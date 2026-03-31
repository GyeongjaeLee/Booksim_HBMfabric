#!/usr/bin/env python3
"""
Run all MoE simulations: structures x bandwidths x scenarios x routings x k x sizes.

Output directory structure:
  RESULT_DIR/{MatrixType}/{Structure}/{Bandwidth}/{Scheme}_{routing}[_nmk{K}_nmp{P}]/k{K}_{Size}MiB[_ir{IR}].txt

============================================================
  Usage examples
============================================================
  # Run specific structure + bandwidth with near-min params and custom injection rate
  python3 run_all_moe.py --structures B100_Global --bandwidths B200+HBM3e \
                         --routings near_min_random near_min_adaptive \
                         --near-min-k 1 2 \
                         --near-min-p 0.0 0.5 1.0 \
                         --injection-rate 0.5 \
                         --matrix-type GPU-HBM

  # Collect results from directories into a single CSV
  python3 run_all_moe.py --collect ./results/moe/GPU-HBM/B100_Global
"""

import os
import sys
import re
import subprocess
import argparse
import csv
import concurrent.futures

from experiments_loader import load_experiments

# ============================================================
#  Configuration
# ============================================================
K_VALUES = [1, 2, 4, 8, 16]
SIZES_MIB = [8, 16, 32, 64, 128, 256]

# Bandwidth: 1 unit = 7 ports
PORTS_PER_UNIT = 7

# Structure and bandwidth tables loaded from experiments.csv
_HERE = os.path.dirname(os.path.abspath(__file__))
STRUCTURES, BANDWIDTHS = load_experiments(os.path.join(_HERE, "experiments.csv"))

# Supported routing functions (for hybrid_routing selection)
ALL_ROUTING_FUNCTIONS = [
    "baseline", "min_oblivious", "min_adaptive",
    "near_min_adaptive", "near_min_random", "valiant", "ugal", "fixed_min"
]

# Default hybrid_routing when --routings not specified
DEFAULT_ROUTINGS = ["near_min_adaptive"]

SCENARIOS = [
    {
        "label": "Baseline",
        "hybrid_scheme": True,
        "is_fabric": 0,
        "matrix_prefix": "moe_matrix_baseline",
    },
    {
        "label": "Fabric",
        "hybrid_scheme": True,
        "is_fabric": 1,
        "matrix_prefix": "moe_matrix_baseline",
    },
    {
        "label": "Offloading",
        "hybrid_scheme": True,
        "is_fabric": 1,
        "matrix_prefix": "moe_matrix_H2H",
    },
]

BASE_CONFIG = "./src/examples/hbmnet_accelsim_config"
MATRIX_DIR = "./src/examples/moe"
RESULT_DIR = "./results/moe"


# ============================================================
#  Helper functions
# ============================================================
def bw_to_overrides(bw_name):
    """Convert bandwidth config to BookSim config overrides."""
    bw = BANDWIDTHS[bw_name]
    return {
        "l2_per_hbm":          str(bw["l2_per_hbm"]),
        "xbar_xbar_bandwidth": str(round(bw["gpu_gpu"] * PORTS_PER_UNIT)),
        "xbar_hbm_bandwidth":  str(round(bw["gpu_hbm"] * PORTS_PER_UNIT)),
        "xbar_mc_bandwidth":   str(round(bw["gpu_hbm"] * PORTS_PER_UNIT)),
        "mc_hbm_bandwidth":    str(round(bw["tsv"]     * PORTS_PER_UNIT)),
        "mc_mc_bandwidth":     str(round(bw["hbm_hbm"] * PORTS_PER_UNIT)),
    }


def struct_to_overrides(struct_name):
    """Convert structure config to BookSim config overrides."""
    s = STRUCTURES[struct_name]
    return {
        "num_xbars":    str(s["num_xbars"]),
        "hbm_per_side": str(s["hbm_per_side"]),
        "sm_per_xbar":  str(s["sm_per_xbar"]),
    }


def scheme_routing_dirname(scenario, routing, nm_k, nm_p):
    """Get directory name for scheme + routing combination."""
    if scenario.get("hybrid_scheme", False):
        base = f"{scenario['label']}_{routing}"
        if routing == "near_min_adaptive":
            base += f"_nmk{nm_k}_nmp{nm_p}"
        elif routing == "near_min_random":
            base += f"_nmk{nm_k}"
        return base
    else:
        return f"{scenario['label']}_baseline"


def get_result_path(result_dir, struct_name, bw_name, sr_dirname, k, size_mib, ir=1.0):
    """Get full path for a result file, appending injection rate if not 1.0."""
    if ir == 1.0:
        filename = f"k{k}_{size_mib}MiB.txt"
    else:
        filename = f"k{k}_{size_mib}MiB_ir{ir}.txt"
    return os.path.join(result_dir, struct_name, bw_name, sr_dirname, filename)


def routings_for_scenario(scenario, active_routings):
    """Get routing list for a scenario. Baseline always uses 'baseline'."""
    if scenario.get("hybrid_scheme", False):
        return active_routings
    else:
        return ["baseline"]


def find_matrix_file(matrix_dir, struct_name, matrix_type, prefix, k, size_mib):
    """Find traffic matrix file inside the MatrixType folder."""
    for ext in [".txt", ""]:
        path = os.path.join(matrix_dir, struct_name, matrix_type, f"{prefix}_k{k}_{size_mib}MiB{ext}")
        if os.path.exists(path):
            return path
    return None


def read_total_mb(matrix_path):
    """Extract moe_total_mb from the sum row of a traffic matrix file."""
    with open(matrix_path, 'r') as f:
        lines = f.readlines()
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if 'sum' in stripped.lower():
            parts = stripped.split('\t')
            for p in reversed(parts):
                p = p.strip()
                try:
                    return float(p)
                except ValueError:
                    continue
    raise ValueError(f"Could not extract moe_total_mb from {matrix_path}")


def generate_config(base_config, overrides, output_path):
    """Generate a BookSim config file from base config with overrides."""
    with open(base_config, 'r') as f:
        lines = f.readlines()
    result = []
    keys_written = set()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('//') or not stripped:
            result.append(line)
            continue
        match = re.match(r'^(\w+)\s*=', stripped)
        if match:
            key = match.group(1)
            if key in overrides:
                result.append(f"{key} = {overrides[key]};\n")
                keys_written.add(key)
                continue
        result.append(line)
    for key, val in overrides.items():
        if key not in keys_written:
            result.append(f"{key} = {val};\n")
    with open(output_path, 'w') as f:
        f.writelines(result)


def extract_results(text):
    """Parse BookSim output text and extract result metrics."""
    r = {}
    def sfloat(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None
    def sint(pattern):
        m = re.search(pattern, text)
        return int(m.group(1)) if m else None

    # Extract injection_rate from config dump
    m_ir = re.search(r'injection_rate\s*=\s*([\d.]+)', text)
    r['injection_rate'] = float(m_ir.group(1)) if m_ir else 1.0

    r['total_mb'] = sfloat(r'moe_total_mb\s*=\s*([\d.]+)')
    r['total_packets'] = sint(r'Total packets\s*=\s*(\d+)')
    r['time_taken'] = sint(r'Time taken is\s*(\d+)\s*cycles')

    r['pkt_lat_avg'] = sfloat(r'Packet latency average\s*=\s*([\d.]+)')
    m = re.search(r'Packet latency average.*?\n\s*minimum\s*=\s*(\d+)', text, re.DOTALL)
    r['pkt_lat_min'] = int(m.group(1)) if m else None
    m = re.search(r'Packet latency average.*?\n.*?maximum\s*=\s*(\d+)', text, re.DOTALL)
    r['pkt_lat_max'] = int(m.group(1)) if m else None

    r['net_lat_avg'] = sfloat(r'Network latency average\s*=\s*([\d.]+)')

    r['flit_lat_avg'] = sfloat(r'Flit latency average\s*=\s*([\d.]+)')
    m = re.search(r'Flit latency average.*?\n\s*minimum\s*=\s*(\d+)', text, re.DOTALL)
    r['flit_lat_min'] = int(m.group(1)) if m else None
    m = re.search(r'Flit latency average.*?\n.*?maximum\s*=\s*(\d+)', text, re.DOTALL)
    r['flit_lat_max'] = int(m.group(1)) if m else None

    r['fragmentation_avg'] = sfloat(r'Fragmentation average\s*=\s*([\d.]+)')
    r['inject_pkt_rate'] = sfloat(r'Injected packet rate average\s*=\s*([\d.]+)')
    r['accept_pkt_rate'] = sfloat(r'Accepted packet rate average\s*=\s*([\d.]+)')
    r['inject_flit_rate'] = sfloat(r'Injected flit rate average\s*=\s*([\d.]+)')
    r['accept_flit_rate'] = sfloat(r'Accepted flit rate average\s*=\s*([\d.]+)')

    r['hops_avg'] = sfloat(r'Hops average\s*=\s*([\d.]+)')
    r['run_time_sec'] = sfloat(r'Total run time\s*([\d.]+)')

    # Escape VC usage
    m = re.search(r'Escape VC usage\s*=\s*(\d+)\s*/\s*(\d+)\s*\(\s*([\d.]+)%\)', text)
    if m:
        r['escape_vc_count'] = int(m.group(1))
        r['escape_vc_total'] = int(m.group(2))
        r['escape_vc_pct']   = float(m.group(3))
    else:
        r['escape_vc_count'] = r['escape_vc_total'] = r['escape_vc_pct'] = None

    # UGAL non-minimal ratio
    m = re.search(r'non-min ratio\s*=\s*([\d.]+)%', text)
    r['ugal_nonmin_pct'] = float(m.group(1)) if m else None
    m = re.search(r'UGAL decisions:\s*total=(\d+)\s+min=(\d+)\s+non-min=(\d+)', text)
    if m:
        r['ugal_total'] = int(m.group(1))
        r['ugal_min']   = int(m.group(2))
        r['ugal_nonmin'] = int(m.group(3))
    else:
        r['ugal_total'] = r['ugal_min'] = r['ugal_nonmin'] = None

    # Near-min adaptive ratio
    m = re.search(r'Near-min decisions:\s*total=(\d+)\s+min=(\d+)\s+non-min=(\d+)', text)
    if m:
        r['nearmin_total']  = int(m.group(1))
        r['nearmin_min']    = int(m.group(2))
        r['nearmin_nonmin'] = int(m.group(3))
    else:
        r['nearmin_total'] = r['nearmin_min'] = r['nearmin_nonmin'] = None
    m = re.search(r'near-min ratio\s*=\s*([\d.]+)%', text)
    r['nearmin_nonmin_pct'] = float(m.group(1)) if m else None

    # Near-min path usage: paths that used near-min at least once / total miss packets
    m = re.search(r'Near-min path usage:\s*(\d+)\s*/\s*(\d+)\s*\(\s*([\d.]+)%\)', text)
    if m:
        r['nearmin_paths_used']  = int(m.group(1))
        r['nearmin_paths_total'] = int(m.group(2))
        r['nearmin_path_pct']    = float(m.group(3))
    else:
        r['nearmin_paths_used'] = r['nearmin_paths_total'] = r['nearmin_path_pct'] = None

    # Miss link avg saturation per type (XBAR_XBAR, XBAR_MC, MC_MC)
    m = re.search(
        r'Miss link avg saturation:\s*XBAR_XBAR=([\d.]+)%\s*XBAR_MC=([\d.]+)%\s*MC_MC=([\d.]+)%',
        text)
    if m:
        r['sat_xbar_xbar'] = float(m.group(1))
        r['sat_xbar_mc']   = float(m.group(2))
        r['sat_mc_mc']     = float(m.group(3))
    else:
        r['sat_xbar_xbar'] = r['sat_xbar_mc'] = r['sat_mc_mc'] = None

    return r


def _fmt_pct(val):
    return f"{val:.1f}%" if val is not None else "N/A"


# ============================================================
#  CSV Fields
# ============================================================
CSV_FIELDS = [
    'structure', 'bandwidth', 'scenario', 'routing_function',
    'nm_k', 'nm_p', 'injection_rate',  # Near-min & Inject parameters
    'k', 'size_mib', 'total_mb',
    'total_packets', 'time_taken',
    'pkt_lat_avg', 'pkt_lat_min', 'pkt_lat_max',
    'net_lat_avg',
    'flit_lat_avg', 'flit_lat_min', 'flit_lat_max',
    'fragmentation_avg',
    'inject_pkt_rate', 'accept_pkt_rate',
    'inject_flit_rate', 'accept_flit_rate',
    'hops_avg', 'run_time_sec',
    'ugal_nonmin_pct', 'ugal_total', 'ugal_min', 'ugal_nonmin',
    'nearmin_nonmin_pct', 'nearmin_total', 'nearmin_min', 'nearmin_nonmin',
    'nearmin_path_pct', 'nearmin_paths_used', 'nearmin_paths_total',
    'sat_xbar_xbar', 'sat_xbar_mc', 'sat_mc_mc',
    'escape_vc_pct', 'escape_vc_count', 'escape_vc_total',
    'matrix_file', 'result_file',
]


# ============================================================
#  Worker Function
# ============================================================
def run_single_scenario(job):
    """Execute a single simulation job."""
    scenario    = job['scenario']
    routing     = job['routing']
    k           = job['k']
    size_mib    = job['size_mib']
    args        = job['args']
    job_num     = job['job_num']
    total_jobs  = job['total_jobs']
    struct_name = job['structure']
    bw_name     = job['bandwidth']
    result_dir  = job['result_dir']

    nm_k        = job['nm_k']
    nm_p        = job['nm_p']
    ir          = args.injection_rate

    sr_dir = scheme_routing_dirname(scenario, routing, nm_k, nm_p)
    result_path = get_result_path(result_dir, struct_name, bw_name, sr_dir, k, size_mib, ir)
    display_path = os.path.relpath(result_path, result_dir)
    matrix_path = find_matrix_file(MATRIX_DIR, struct_name, args.matrix_type, scenario["matrix_prefix"], k, size_mib)
    base_label = scenario["label"]

    # ── Parse-only mode ──
    if args.parse_only:
        if not os.path.exists(result_path):
            return {'status': 'skipped', 'row': None,
                    'msg': f"[{job_num}/{total_jobs}] [MISSING] {display_path}"}
        with open(result_path, 'r') as f:
            text = f.read()
        r = extract_results(text)
        total_mb = r.get('total_mb', "")
        if not total_mb and matrix_path:
            try: total_mb = read_total_mb(matrix_path)
            except: pass
        row = _make_row(struct_name, bw_name, base_label, routing, nm_k, nm_p, k, size_mib,
                        total_mb, matrix_path or '', result_path, r)
        return {'status': 'parsed', 'row': row,
                'msg': f"[{job_num}/{total_jobs}] [PARSED] {display_path}"}

    # ── Check matrix file ──
    if matrix_path is None:
        return {'status': 'skipped', 'row': None,
                'msg': f"[{job_num}/{total_jobs}] [SKIP] Matrix not found: "
                       f"{scenario['matrix_prefix']}_k{k}_{size_mib}MiB in {struct_name}/{args.matrix_type}"}

    # ── Skip-existing mode ──
    if args.skip_existing and os.path.exists(result_path):
        with open(result_path, 'r') as f:
            text = f.read()
        r = extract_results(text)
        total_mb = r.get('total_mb', "")
        if not total_mb:
            try: total_mb = read_total_mb(matrix_path)
            except: pass
        row = _make_row(struct_name, bw_name, base_label, routing, nm_k, nm_p, k, size_mib,
                        total_mb, matrix_path, result_path, r)
        return {'status': 'skipped', 'row': row,
                'msg': f"[{job_num}/{total_jobs}] [EXISTS] {display_path}"}

    # ── Read total MB ──
    try:
        total_mb = read_total_mb(matrix_path)
    except ValueError as e:
        return {'status': 'failed', 'row': None,
                'msg': f"[{job_num}/{total_jobs}] [ERROR] {e}"}

    # ── Build config overrides ──
    overrides = {
        "is_fabric": str(scenario["is_fabric"]),
        "traffic_matrix_file": matrix_path,
        "moe_total_mb": str(total_mb),
        "injection_rate": str(ir),
        "routing_function": "hybrid",
        "hybrid_routing": routing,
        "sim_type": "moe_accelsim",           # Added for MoE simulation
        "traffic": "uniform",                 # Added for MoE simulation
        "injection_process": "bernoulli"      # Added for MoE simulation
    }
    
    overrides.update(struct_to_overrides(struct_name))
    overrides.update(bw_to_overrides(bw_name))

    if routing == "near_min_adaptive":
        overrides["near_min_k"] = str(nm_k)
        overrides["near_min_penalty"] = str(nm_p)
    elif routing == "near_min_random":
        overrides["near_min_k"] = str(nm_k)

    # ── Write temp config & run ──
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    tmp_cfg = result_path + f".tmp_{job_num}.cfg"
    generate_config(BASE_CONFIG, overrides, tmp_cfg)

    cmd = [args.booksim, tmp_cfg]
    header_msg = (f"[{job_num}/{total_jobs}] [RUN] {display_path} "
                  f"total_mb={total_mb:.4f}")

    if args.dry_run:
        if os.path.exists(tmp_cfg):
            os.remove(tmp_cfg)
        return {'status': 'skipped', 'row': None,
                'msg': f"{header_msg}\n      cmd: {' '.join(cmd)}"}

    status = 'failed'
    row = None
    msg = header_msg
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        text = proc.stdout + proc.stderr
        with open(result_path, 'w') as f:
            f.write(text)
        r = extract_results(text)
        row = _make_row(struct_name, bw_name, base_label, routing, nm_k, nm_p, k, size_mib,
                        total_mb, matrix_path, result_path, r)
        tp = r.get('total_packets')
        tt = r.get('time_taken')
        lat = r.get('pkt_lat_avg')
        status = 'succeeded' if tp else 'failed'
        msg = (f"{header_msg}\n"
               f"      [{'OK' if tp else 'FAIL'}] time={tt} packets={tp} latency={lat} hops={r.get('hops_avg')}"
               f"  NearMin={_fmt_pct(r.get('nearmin_nonmin_pct'))}"
               f"  NMPath={_fmt_pct(r.get('nearmin_path_pct'))}"
               f"  EscVC={_fmt_pct(r.get('escape_vc_pct'))}")
    except subprocess.TimeoutExpired:
        msg = f"{header_msg}\n      [TIMEOUT]"
    except Exception as e:
        msg = f"{header_msg}\n      [ERROR] {e}"
    finally:
        if os.path.exists(tmp_cfg):
            os.remove(tmp_cfg)

    return {'status': status, 'row': row, 'msg': msg}


def _make_row(struct_name, bw_name, scenario_label, routing, nm_k, nm_p, k, size_mib,
              total_mb, matrix_path, result_path, parsed):
    """Build a CSV row dict from parsed results."""
    eff_nm_p = "" if routing == "near_min_random" else nm_p
    eff_nm_k = nm_k if "near_min" in routing else ""

    row = {
        'structure': struct_name, 'bandwidth': bw_name,
        'scenario': scenario_label, 'routing_function': routing,
        'nm_k': eff_nm_k, 'nm_p': eff_nm_p,
        'injection_rate': parsed.get('injection_rate', 1.0),
        'k': k, 'size_mib': size_mib, 'total_mb': total_mb,
        'matrix_file': matrix_path, 'result_file': result_path,
    }
    row.update({k2: v for k2, v in parsed.items() if v is not None})
    return row


def _sort_key(row):
    struct_order = {name: i for i, name in enumerate(STRUCTURES.keys())}
    bw_order = {name: i for i, name in enumerate(BANDWIDTHS.keys())}
    scenario_order = {'Baseline': 0, 'Fabric': 1, 'Offloading': 2}
    
    nm_k = row.get('nm_k', 0)
    nm_k_val = 0 if nm_k in ('', None) else int(nm_k)
    
    nm_p = row.get('nm_p', 0.0)
    nm_p_val = 0.0 if nm_p in ('', None) else float(nm_p)

    ir = row.get('injection_rate', 1.0)
    ir_val = 1.0 if ir in ('', None) else float(ir)
    
    return (
        struct_order.get(row.get('structure', ''), 999),
        bw_order.get(row.get('bandwidth', ''), 999),
        scenario_order.get(row.get('scenario', ''), 999),
        row.get('routing_function', ''),
        nm_k_val,
        nm_p_val,
        ir_val,
        row.get('k', 0),
        row.get('size_mib', 0),
    )


def collect_results(dirs, csv_path):
    """Walk directories, parse all k{K}_{Size}MiB[_ir{IR}].txt files, produce CSV."""
    rows = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"[WARNING] Not a directory: {d}")
            continue
        for root, _, files in os.walk(d):
            for fname in sorted(files):
                if not fname.endswith('.txt') or fname.startswith('.'):
                    continue
                # Match k{K}_{Size}MiB.txt OR k{K}_{Size}MiB_ir{IR}.txt
                m = re.match(r'k(\d+)_(\d+)MiB(?:_ir([\d.]+))?\.txt$', fname)
                if not m:
                    continue
                fpath = os.path.join(root, fname)
                k_val = int(m.group(1))
                size_mib = int(m.group(2))

                parts = os.path.normpath(fpath).split(os.sep)
                sr_name     = parts[-2] if len(parts) >= 2 else ''
                bw_name     = parts[-3] if len(parts) >= 3 else ''
                struct_name = parts[-4] if len(parts) >= 4 else ''

                scheme, routing = '', sr_name
                nm_k, nm_p = "", ""
                for s in ['Offloading', 'Baseline', 'Fabric']:
                    if sr_name.startswith(s + '_'):
                        scheme = s
                        rest = sr_name[len(s) + 1:]
                        
                        m_adp = re.match(r'^(near_min_adaptive)_nmk(\d+)_nmp([\d.]+)$', rest)
                        m_rnd = re.match(r'^(near_min_random)_nmk(\d+)$', rest)
                        
                        if m_adp:
                            routing = m_adp.group(1)
                            nm_k = int(m_adp.group(2))
                            nm_p = float(m_adp.group(3))
                        elif m_rnd:
                            routing = m_rnd.group(1)
                            nm_k = int(m_rnd.group(2))
                            nm_p = ""
                        else:
                            routing = rest
                        break

                with open(fpath, 'r') as f:
                    text = f.read()
                r = extract_results(text)
                total_mb = r.get('total_mb', '')
                row = {
                    'structure': struct_name, 'bandwidth': bw_name,
                    'scenario': scheme, 'routing_function': routing,
                    'nm_k': nm_k, 'nm_p': nm_p,
                    'k': k_val, 'size_mib': size_mib, 'total_mb': total_mb,
                    'result_file': fpath,
                }
                row.update({k2: v for k2, v in r.items() if v is not None})
                rows.append(row)

    rows.sort(key=_sort_key)

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Collected {len(rows)} results -> {csv_path}")

    if rows:
        _print_result_table(rows)

    return rows


def print_plan_summary(active_structs, active_bws, active_scenarios, active_routings,
                       active_k_values, active_sizes, result_dir, args):
    print()
    print("=" * 130)
    print("  EXECUTION PLAN")
    print("=" * 130)

    mode_flags = []
    if args.dry_run:       mode_flags.append("DRY-RUN")
    if args.parse_only:    mode_flags.append("PARSE-ONLY")
    if args.skip_existing: mode_flags.append("SKIP-EXISTING")
    if mode_flags:
        print(f"  Mode: {', '.join(mode_flags)}")

    print()
    print("  Bandwidth port mappings (1 unit = 7 ports):")
    for bw_name in active_bws:
        ports = bw_to_overrides(bw_name)
        print(f"    {bw_name:<18} xbar_xbar={ports['xbar_xbar_bandwidth']:>3}  "
              f"xbar_hbm={ports['xbar_hbm_bandwidth']:>3}  xbar_mc={ports['xbar_mc_bandwidth']:>3}  "
              f"mc_hbm={ports['mc_hbm_bandwidth']:>3}  mc_mc={ports['mc_mc_bandwidth']:>3}")
    print()

    hdr = (f"  {'#':>4}  {'Structure':<18} {'Bandwidth':<18} {'Scenario':<12} "
           f"{'Routing':<25} {'IR':>4} {'k':>3} {'Size':>6}  {'Matrix':^8}  {'Result':^8}  {'MatrixFile'}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    counts = {'new': 0, 'exists': 0, 'no_matrix': 0}
    job_num = 0
    for struct_name in active_structs:
        for bw_name in active_bws:
            for scenario in active_scenarios:
                routings = routings_for_scenario(scenario, active_routings)
                for routing in routings:
                    if routing == "near_min_adaptive":
                        k_list = args.near_min_k
                        p_list = args.near_min_p
                    elif routing == "near_min_random":
                        k_list = args.near_min_k
                        p_list = [""]
                    else:
                        k_list = [""]
                        p_list = [""]
                        
                    for nm_k in k_list:
                        for nm_p in p_list:
                            sr_dir = scheme_routing_dirname(scenario, routing, nm_k, nm_p)
                            
                            disp_routing = routing
                            if routing == "near_min_adaptive":
                                disp_routing += f" (k={nm_k}, p={nm_p})"
                            elif routing == "near_min_random":
                                disp_routing += f" (k={nm_k})"

                            for k in active_k_values:
                                for size_mib in active_sizes:
                                    job_num += 1
                                    result_path = get_result_path(result_dir, struct_name, bw_name,
                                                                  sr_dir, k, size_mib, args.injection_rate)
                                    matrix_path = find_matrix_file(MATRIX_DIR, struct_name, args.matrix_type, scenario["matrix_prefix"], k, size_mib)

                                    if matrix_path is None:
                                        matrix_tag = "[MISS]"
                                        result_tag = "  -   "
                                        counts['no_matrix'] += 1
                                        matrix_display = (f"{scenario['matrix_prefix']}_k{k}"
                                                          f"_{size_mib}MiB (NOT FOUND)")
                                    else:
                                        matrix_tag = "[ OK ]"
                                        matrix_display = os.path.basename(matrix_path)
                                        if os.path.exists(result_path):
                                            result_tag = "[EXISTS]"
                                            counts['exists'] += 1
                                        else:
                                            result_tag = "[ NEW ]"
                                            counts['new'] += 1

                                    print(f"  {job_num:>4}  {struct_name:<18} {bw_name:<18} "
                                          f"{scenario['label']:<12} {disp_routing:<25} {args.injection_rate:>4.2f} {k:>3} "
                                          f"{str(size_mib)+'MiB':>6}  {matrix_tag}  {result_tag}  "
                                          f"{matrix_display}")

    print()
    total = counts['new'] + counts['exists'] + counts['no_matrix']
    print(f"  Total jobs : {total}")
    print(f"    [ NEW ]  : {counts['new']}  (will run)")
    print(f"  [EXISTS]   : {counts['exists']}  ({'skip' if args.skip_existing else 'overwrite'})")
    print(f"  [NO MATRIX]: {counts['no_matrix']}  (will skip)")
    print("=" * 130)
    print()


def _print_result_table(rows):
    """Print a formatted result summary table."""
    print()
    hdr = (f"{'Structure':<18} {'Bandwidth':<18} {'Scenario':<12} {'Routing':<18} "
           f"{'IR':>4} {'k':>3} {'Size':>6} {'Packets':>10} {'Cycles':>12} {'AvgLat':>9} {'Hops':>6} "
           f"{'NM%':>6} {'NMPath%':>8} "
           f"{'XX_sat':>7} {'XMC_sat':>8} {'MC_sat':>7} {'EscVC%':>7}")
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        def _s(key, default=''):
            v = row.get(key, default)
            return '' if v is None else str(v)
        
        ir_str = f"{float(row.get('injection_rate', 1.0)):.2f}"
        
        print(f"{_s('structure'):<18} {_s('bandwidth'):<18} {_s('scenario'):<12} "
              f"{_s('routing_function'):<18} {ir_str:>4} {_s('k'):>3} {_s('size_mib')+'M':>6} "
              f"{_s('total_packets'):>10} {_s('time_taken'):>12} {_s('pkt_lat_avg'):>9} {_s('hops_avg'):>6} "
              f"{_fmt_pct(row.get('nearmin_nonmin_pct')):>6} "
              f"{_fmt_pct(row.get('nearmin_path_pct')):>8} "
              f"{_fmt_pct(row.get('sat_xbar_xbar')):>7} "
              f"{_fmt_pct(row.get('sat_xbar_mc')):>8} "
              f"{_fmt_pct(row.get('sat_mc_mc')):>7} "
              f"{_fmt_pct(row.get('escape_vc_pct')):>7}")


def main():
    global RESULT_DIR
    parser = argparse.ArgumentParser(
        description="Run all MoE booksim simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Structures:  " + ", ".join(STRUCTURES.keys()),
            "Bandwidths:  " + ", ".join(BANDWIDTHS.keys()),
            "Routings:    " + ", ".join(ALL_ROUTING_FUNCTIONS),
            "",
            "Output: RESULT_DIR/{MatrixType}/{Structure}/{Bandwidth}/{Scheme}_{routing}[_nmk{K}_nmp{P}]/k{K}_{Size}MiB[_ir{IR}].txt",
        ])
    )
    parser.add_argument('--dry-run', action='store_true', help='Print commands only')
    parser.add_argument('--skip-existing', action='store_true', help='Skip if result exists')
    parser.add_argument('--parse-only', action='store_true', help='Only parse existing results')
    parser.add_argument('--booksim', default='./src/booksim', help='Path to booksim binary')
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt')
    parser.add_argument('--result-dir', default=RESULT_DIR,
                        help=f'Result output directory (default: {RESULT_DIR})')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Subprocess timeout in seconds (default: None, wait indefinitely)')

    parser.add_argument('--structures', nargs='+', metavar='STRUCT', default=None)
    parser.add_argument('--bandwidths', nargs='+', metavar='BW', default=None)
    parser.add_argument('--schemes', nargs='+', metavar='SCHEME', default=None)
    parser.add_argument('--routings', nargs='+', metavar='ROUTING', default=None)
    parser.add_argument('--k-values', nargs='+', type=int, default=None)
    parser.add_argument('--sizes', nargs='+', type=int, default=None)
    
    parser.add_argument('--near-min-k', nargs='+', type=int, default=[2])
    parser.add_argument('--near-min-p', nargs='+', type=float, default=[1.0])
    parser.add_argument('--injection-rate', type=float, default=1.0,
                        help='Injection rate for MoE traffic (default: 1.0)')
    parser.add_argument('--matrix-type', choices=['GPU-HBM', 'End-to-End'], default='GPU-HBM')

    parser.add_argument('--collect', nargs='+', metavar='DIR')
    parser.add_argument('--collect-output', metavar='CSV', default=None)

    args = parser.parse_args()
    RESULT_DIR = os.path.join(args.result_dir, args.matrix_type)

    if args.collect:
        csv_path = args.collect_output or os.path.join(RESULT_DIR, "collected.csv")
        collect_results(args.collect, csv_path)
        return

    os.makedirs(RESULT_DIR, exist_ok=True)

    active_structs = list(STRUCTURES.keys())
    if args.structures:
        active_structs = args.structures

    active_bws = list(BANDWIDTHS.keys())
    if args.bandwidths:
        active_bws = args.bandwidths

    active_scenarios = SCENARIOS
    if args.schemes:
        wanted = [s.lower() for s in args.schemes]
        active_scenarios = [s for s in SCENARIOS if s["label"].lower() in wanted]

    active_routings = args.routings if args.routings else DEFAULT_ROUTINGS
    active_k = args.k_values or K_VALUES
    active_sizes = args.sizes or SIZES_MIB

    jobs = []
    job_num = 0
    for struct_name in active_structs:
        for bw_name in active_bws:
            for scenario in active_scenarios:
                routings = routings_for_scenario(scenario, active_routings)
                for routing in routings:
                    if routing == "near_min_adaptive":
                        k_list = args.near_min_k
                        p_list = args.near_min_p
                    elif routing == "near_min_random":
                        k_list = args.near_min_k
                        p_list = [""]
                    else:
                        k_list = [""]
                        p_list = [""]
                        
                    for nm_k in k_list:
                        for nm_p in p_list:
                            for k in active_k:
                                for size_mib in active_sizes:
                                    job_num += 1
                                    jobs.append({
                                        'structure': struct_name,
                                        'bandwidth': bw_name,
                                        'scenario': scenario,
                                        'routing': routing,
                                        'nm_k': nm_k,
                                        'nm_p': nm_p,
                                        'k': k,
                                        'size_mib': size_mib,
                                        'args': args,
                                        'result_dir': RESULT_DIR,
                                        'job_num': job_num,
                                        'total_jobs': 0,
                                    })

    total_jobs = len(jobs)
    for j in jobs:
        j['total_jobs'] = total_jobs

    print_plan_summary(active_structs, active_bws, active_scenarios, active_routings,
                       active_k, active_sizes, RESULT_DIR, args)

    if not args.yes and not args.parse_only and not args.dry_run:
        try:
            input("Press Enter to start, or Ctrl-C to cancel... ")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
        print()

    csv_rows = []
    succeeded, skipped, failed = 0, 0, 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_scenario, job): job for job in jobs}
        try:
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                print(res['msg'])
                st = res['status']
                if st == 'succeeded':  succeeded += 1
                elif st == 'failed':   failed += 1
                elif st in ['skipped', 'parsed']: skipped += 1
                if res['row']:
                    csv_rows.append(res['row'])
        except KeyboardInterrupt:
            print("\n[WARNING] Interrupted! Shutting down...")
            executor.shutdown(wait=False, cancel_futures=True)
            sys.exit(1)

    csv_rows.sort(key=_sort_key)
    csv_path = os.path.join(RESULT_DIR, "summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    if csv_rows:
        _print_result_table(csv_rows)

if __name__ == '__main__':
    main()