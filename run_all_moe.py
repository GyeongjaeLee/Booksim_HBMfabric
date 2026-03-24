#!/usr/bin/env python3
"""
Run all MoE simulations: scenarios × routing functions × k values × sizes (PARALLEL VERSION).

============================================================
  Available routing functions for hbmnet
============================================================
  "baseline"      Minimal routing.
                  Recommended for fabric=0 (no HBM-HBM links).

  "min_oblivious" Minimal oblivious routing using all VCs.
                  No adaptive port selection; low overhead.

  "min_adaptive"  Minimal adaptive: picks least-congested output port.
                  Stays on shortest path; good for light imbalance.

  "valiant"       Valiant oblivious: routes through a random intermediate
                  router before destination (non-minimal, full load balance).

  "ugal"          UGAL (Universal Globally Adaptive Load-balancing).
                  Compares minimal vs non-minimal cost using ugal_threshold;
                  adapts between direct and detour paths per packet.
                  Reports UGAL non-minimal ratio in output.

  "hybrid"        Hybrid routing (behavior controlled by 'hybrid_routing'
                  config key in the base config file).

============================================================
  Scenario definitions (SCENARIOS list)
============================================================
  Each scenario dict:
    "label"            : Unique name used in output filenames and CSV.
    "routing_function" : Default routing for this scenario.
                         Override per-run with --routings.
    "is_fabric"        : 1 = enable HBM-HBM fabric links.
                         0 = baseline topology only (baseline routing recommended).
    "matrix_prefix"    : Prefix for traffic matrix files in MATRIX_DIR.

  To add a new scenario, append a dict to SCENARIOS:
    {
        "label": "MyScenario",
        "routing_function": "ugal",   # choose from routing functions above
        "is_fabric": 1,
        "matrix_prefix": "moe_matrix_custom",
    }

============================================================
  Usage examples
============================================================
  # Run all default scenarios
  python3 run_all_moe.py

  # Compare multiple routing functions (creates label_routing variants)
  python3 run_all_moe.py --routings ugal valiant min_adaptive --schemes Fabric

  # Override routing for specific scheme (single routing = no variant suffix)
  python3 run_all_moe.py --routings ugal --schemes Fabric Offloading

  # Filter by k value and size
  python3 run_all_moe.py --k-values 1 4 --sizes 8 64

  # Other options
  [--dry-run] [--skip-existing] [--parse-only]
  [--booksim ./src/booksim] [--workers 8]
  [--schemes Baseline Fabric Offloading]
  [--k-values 1 2 4 8 16]
  [--sizes 8 16 32 64 128 256]
  [--routings baseline min_oblivious min_adaptive valiant ugal hybrid]
"""

import os
import sys
import re
import subprocess
import argparse
import csv
import concurrent.futures

# ============================================================
#  Configuration
# ============================================================
K_VALUES = [1, 2, 4, 8, 16]
SIZES_MIB = [8, 16, 32, 64, 128, 256]

# All supported routing functions (for --routings validation)
ALL_ROUTING_FUNCTIONS = [
    "baseline",       # minimal, escape VC only — use with is_fabric=0
    "min_oblivious",  # minimal oblivious, all VCs
    "min_adaptive",   # minimal adaptive (least-loaded port)
    "valiant",        # Valiant oblivious (random intermediate)
    "ugal",           # UGAL adaptive (min vs non-min cost comparison)
    "hybrid",         # hybrid (see hybrid_routing config key)
]

SCENARIOS = [
    # ──────────────────────────────────────────────────────────
    # Baseline: no fabric links, baseline (escape-only) routing.
    # Use routing_function="baseline".
    # ──────────────────────────────────────────────────────────
    {
        "label": "Baseline",
        "routing_function": "baseline",
        "is_fabric": 0,
        "matrix_prefix": "moe_matrix_baseline",
    },
    # ──────────────────────────────────────────────────────────
    # Fabric: HBM-HBM fabric links enabled.
    # Can also compare: min_oblivious, min_adaptive, valiant, ugal hybrid.
    # ──────────────────────────────────────────────────────────
    {
        "label": "Fabric",
        "routing_function": "hybrid",       # UGAL adaptive (default for fabric)
        "is_fabric": 1,
        "matrix_prefix": "moe_matrix_baseline",
    },
    # ──────────────────────────────────────────────────────────
    # Offloading: fabric on + HBM-to-HBM traffic matrix.
    # Measures benefit of offloading SM→L2 traffic to HBM fabric.
    # ──────────────────────────────────────────────────────────
    {
        "label": "Offloading",
        "routing_function": "hybrid",       # UGAL adaptive (default for fabric)
        "is_fabric": 1,
        "matrix_prefix": "moe_matrix_H2H",
    },
]

BASE_CONFIG = "./src/examples/hbmnet_accelsim_config"
MATRIX_DIR = "./src/examples/end-to-end/"
RESULT_DIR = "./results-test"


# ============================================================
#  Helper functions
# ============================================================
def find_matrix_file(prefix, k, size_mib):
    for ext in [".txt", ""]:
        path = os.path.join(MATRIX_DIR, f"{prefix}_k{k}_{size_mib}MiB{ext}")
        if os.path.exists(path):
            return path
    return None

def read_total_mb(matrix_path):
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
    r = {}
    def sfloat(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None
    def sint(pattern):
        m = re.search(pattern, text)
        return int(m.group(1)) if m else None

    r['total_mb'] = sfloat(r'moe_total_mb\s*=\s*([\d.]+)')
    r['total_packets'] = sint(r'Total packets\s*=\s*(\d+)')
    r['completion_time'] = sint(r'MoE(?:AccelSim)? Batch Completion Time\s*=\s*(\d+)')
    r['time_taken'] = sint(r'Time taken is\s*(\d+)\s*cycles')
    r['moe_avg_pkt_latency'] = sfloat(r'Avg packet latency\s*=\s*([\d.]+)')
    r['moe_avg_net_latency'] = sfloat(r'Avg network latency\s*=\s*([\d.]+)')
    r['moe_avg_hops'] = sfloat(r'Avg hops\s*=\s*([\d.]+)')

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

    # UGAL non-minimal path ratio
    # Output format: "  UGAL decisions: total=N  min=M  non-min=P  non-min ratio=X.XX%"
    m = re.search(r'non-min ratio\s*=\s*([\d.]+)%', text)
    r['ugal_nonmin_pct'] = float(m.group(1)) if m else None
    m = re.search(r'UGAL decisions:\s*total=(\d+)\s+min=(\d+)\s+non-min=(\d+)', text)
    if m:
        r['ugal_total'] = int(m.group(1))
        r['ugal_min']   = int(m.group(2))
        r['ugal_nonmin'] = int(m.group(3))
    else:
        r['ugal_total'] = r['ugal_min'] = r['ugal_nonmin'] = None

    # Escape VC usage ratio
    # Output format: "  Escape VC usage = N / M  (X.XX%)"
    m = re.search(r'Escape VC usage\s*=\s*(\d+)\s*/\s*(\d+)\s*\(\s*([\d.]+)%\)', text)
    if m:
        r['escape_vc_count'] = int(m.group(1))
        r['escape_vc_total'] = int(m.group(2))
        r['escape_vc_pct']   = float(m.group(3))
    else:
        r['escape_vc_count'] = r['escape_vc_total'] = r['escape_vc_pct'] = None

    return r


# ============================================================
#  Worker Function (Parallel Execution Target)
# ============================================================
def run_single_scenario(job_info):
    scenario, routing, k, size_mib, args, use_routing_suffix, job_num, total_jobs = job_info
    base_label = scenario["label"]
    # When comparing multiple routings, append routing name to label
    label = f"{base_label}_{routing}" if use_routing_suffix else base_label

    size_dir = os.path.join(RESULT_DIR, f"{size_mib}MiB")
    os.makedirs(size_dir, exist_ok=True)
    result_name = f"{label}_k{k}_{size_mib}MiB.txt"
    result_path = os.path.join(size_dir, result_name)
    matrix_path = find_matrix_file(scenario["matrix_prefix"], k, size_mib)

    row = None
    msg = ""

    if args.parse_only:
        if not os.path.exists(result_path):
            return {'status': 'skipped', 'row': None, 'msg': f"[{job_num}/{total_jobs}] [MISSING] {result_name}"}

        with open(result_path, 'r') as f:
            text = f.read()
        r = extract_results(text)
        total_mb = r.get('total_mb', "")
        if not total_mb and matrix_path:
            try: total_mb = read_total_mb(matrix_path)
            except: pass

        row = {'job_num': job_num, 'scenario': base_label, 'routing_function': routing, 'k': k, 'size_mib': size_mib,
               'total_mb': total_mb, 'matrix_file': matrix_path or '',
               'result_file': result_path}
        row.update({k2: v for k2, v in r.items() if v is not None})
        ct = r.get('completion_time')
        msg = (f"[{job_num}/{total_jobs}] [{'OK' if ct else 'INCOMPLETE'}] {result_name}"
               f"  completion={ct}  lat={r.get('moe_avg_pkt_latency')}"
               f"  ugal_nonmin={_fmt_pct(r.get('ugal_nonmin_pct'))}"
               f"  escape_vc={_fmt_pct(r.get('escape_vc_pct'))}")
        return {'status': 'parsed', 'row': row, 'msg': msg}

    if matrix_path is None:
        return {'status': 'skipped', 'row': None,
                'msg': f"[{job_num}/{total_jobs}] [SKIP] Matrix not found: {scenario['matrix_prefix']}_k{k}_{size_mib}MiB"}

    if args.skip_existing and os.path.exists(result_path):
        with open(result_path, 'r') as f:
            text = f.read()
        r = extract_results(text)
        total_mb = r.get('total_mb', "")
        if not total_mb:
            try: total_mb = read_total_mb(matrix_path)
            except: pass

        row = {'job_num': job_num, 'scenario': base_label, 'routing_function': routing, 'k': k, 'size_mib': size_mib,
               'total_mb': total_mb, 'matrix_file': matrix_path,
               'result_file': result_path}
        row.update({k2: v for k2, v in r.items() if v is not None})
        return {'status': 'skipped', 'row': row, 'msg': f"[{job_num}/{total_jobs}] [EXISTS] {result_name}"}

    try:
        total_mb = read_total_mb(matrix_path)
    except ValueError as e:
        return {'status': 'failed', 'row': None, 'msg': f"[{job_num}/{total_jobs}] [ERROR] {e}"}

    tmp_cfg = os.path.join(size_dir, f"_tmp_{label}_k{k}_{job_num}.cfg")
    overrides = {
        "routing_function": routing,
        "is_fabric": str(scenario["is_fabric"]),
        "traffic_matrix_file": matrix_path,
        "moe_total_mb": str(total_mb),
    }
    generate_config(BASE_CONFIG, overrides, tmp_cfg)

    cmd = [args.booksim, tmp_cfg]
    header_msg = f"[{job_num}/{total_jobs}] [RUN] {label} k={k} size={size_mib}MiB total_mb={total_mb:.4f}"

    if args.dry_run:
        if os.path.exists(tmp_cfg):
            os.remove(tmp_cfg)
        return {'status': 'skipped', 'row': None, 'msg': f"{header_msg}\n      cmd: {' '.join(cmd)}"}

    status = 'failed'
    row = None

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        text = proc.stdout + proc.stderr

        with open(result_path, 'w') as f:
            f.write(text)

        r = extract_results(text)
        row = {'job_num': job_num, 'scenario': base_label, 'routing_function': routing, 'k': k, 'size_mib': size_mib,
               'total_mb': total_mb, 'matrix_file': matrix_path,
               'result_file': result_path}
        # Only update with non-None values to avoid overwriting pre-set fields (e.g. total_mb)
        row.update({k2: v for k2, v in r.items() if v is not None})

        ct = r.get('completion_time')
        al = r.get('moe_avg_pkt_latency')
        status = 'succeeded' if ct else 'failed'
        msg = (f"{header_msg}\n"
               f"      [{'OK' if ct else 'FAIL'}] completion={ct} latency={al} hops={r.get('moe_avg_hops')}\n"
               f"      UGAL non-min={_fmt_pct(r.get('ugal_nonmin_pct'))}  "
               f"Escape VC={_fmt_pct(r.get('escape_vc_pct'))}")

    except subprocess.TimeoutExpired:
        msg = f"{header_msg}\n      [TIMEOUT]"
    except Exception as e:
        msg = f"{header_msg}\n      [ERROR] {e}"
    finally:
        if os.path.exists(tmp_cfg):
            os.remove(tmp_cfg)

    return {'status': status, 'row': row, 'msg': msg}


def _fmt_pct(val):
    return f"{val:.1f}%" if val is not None else "N/A"


# ============================================================
#  Routing Comparison Table
# ============================================================
def print_routing_comparison(csv_rows, routings):
    """Print a side-by-side comparison table grouped by (scenario, k, size_mib)."""
    if len(routings) <= 1:
        return

    from collections import defaultdict
    groups = defaultdict(dict)
    for row in csv_rows:
        key = (row.get('scenario', ''), row.get('k', ''), row.get('size_mib', ''))
        rf  = row.get('routing_function', '')
        groups[key][rf] = row

    print()
    print("=" * 100)
    print("  ROUTING COMPARISON")
    print("=" * 100)
    hdr_cols = ["Scenario", "k", "Size", "Routing", "Completion", "AvgLat", "Hops",
                "UGAL non-min%", "EscapeVC%"]
    widths   = [13, 4, 6, 15, 11, 9, 6, 14, 10]
    hdr = "  ".join(f"{h:<{w}}" for h, w in zip(hdr_cols, widths))
    print(hdr)
    print("-" * len(hdr))

    for key in sorted(groups.keys()):
        scenario, size_mib, k = key
        first = True
        for rf in routings:
            row = groups[key].get(rf)
            if row is None:
                continue
            s_label = scenario if first else ""
            k_label  = str(k) if first else ""
            sz_label = f"{size_mib}MiB" if first else ""
            first = False
            vals = [
                s_label, k_label, sz_label, rf,
                str(row.get('completion_time', '')),
                str(row.get('moe_avg_pkt_latency', '')),
                str(row.get('moe_avg_hops', '')),
                _fmt_pct(row.get('ugal_nonmin_pct')),
                _fmt_pct(row.get('escape_vc_pct')),
            ]
            print("  ".join(f"{v:<{w}}" for v, w in zip(vals, widths)))
        if not first:
            print()


# ============================================================
#  Plan Summary
# ============================================================
def print_plan_summary(active_scenarios, active_routings, active_k_values, active_sizes,
                       use_routing_suffix, result_dir, args):
    """Print a table of all planned jobs before execution starts."""
    print()
    print("=" * 100)
    print("  EXECUTION PLAN")
    print("=" * 100)

    mode_flags = []
    if args.dry_run:    mode_flags.append("DRY-RUN")
    if args.parse_only: mode_flags.append("PARSE-ONLY")
    if args.skip_existing: mode_flags.append("SKIP-EXISTING")
    if mode_flags:
        print(f"  Mode: {', '.join(mode_flags)}")

    # Column header
    hdr = f"  {'#':>4}  {'Scenario':<13} {'Routing':<15} {'k':>3} {'Size':>6}  {'Matrix':^8}  {'Result':^8}  {'MatrixFile'}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    counts = {'new': 0, 'exists': 0, 'no_matrix': 0}
    job_num = 0
    for scenario in active_scenarios:
        routings_for_scenario = active_routings if active_routings else [scenario["routing_function"]]
        for routing in routings_for_scenario:
            for k in active_k_values:
                for size_mib in active_sizes:
                    job_num += 1
                    label = (f"{scenario['label']}_{routing}" if use_routing_suffix
                             else scenario["label"])
                    result_name = f"{label}_k{k}_{size_mib}MiB.txt"
                    result_path = os.path.join(result_dir, f"{size_mib}MiB", result_name)
                    matrix_path = find_matrix_file(scenario["matrix_prefix"], k, size_mib)

                    if matrix_path is None:
                        matrix_tag = "[MISS]"
                        result_tag = "  -   "
                        counts['no_matrix'] += 1
                        matrix_display = f"{scenario['matrix_prefix']}_k{k}_{size_mib}MiB  (NOT FOUND)"
                    else:
                        matrix_tag = "[ OK ]"
                        matrix_display = os.path.basename(matrix_path)
                        if os.path.exists(result_path):
                            result_tag = "[EXISTS]"
                            counts['exists'] += 1
                        else:
                            result_tag = "[ NEW ]"
                            counts['new'] += 1

                    print(f"  {job_num:>4}  {scenario['label']:<13} {routing:<15} {k:>3} "
                          f"{str(size_mib)+'MiB':>6}  {matrix_tag}  {result_tag}  {matrix_display}")

    print()
    total = counts['new'] + counts['exists'] + counts['no_matrix']
    print(f"  Total jobs : {total}")
    print(f"    [ NEW ]  : {counts['new']}  (will run)")
    print(f"  [EXISTS]   : {counts['exists']}  ({'skip' if args.skip_existing else 'overwrite'})")
    print(f"  [NO MATRIX]: {counts['no_matrix']}  (will skip)")
    print("=" * 100)
    print()


# ============================================================
#  Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run all MoE booksim simulations in Parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Routing functions: " + ", ".join(ALL_ROUTING_FUNCTIONS),
            "  baseline     - minimal, escape VC only (recommended for is_fabric=0)",
            "  min_oblivious- minimal oblivious, all VCs",
            "  min_adaptive - minimal adaptive (least-loaded port)",
            "  valiant      - Valiant oblivious (random intermediate router)",
            "  ugal         - UGAL adaptive (min vs non-min cost, ugal_threshold)",
            "  hybrid       - hybrid (see hybrid_routing in base config)",
        ])
    )
    parser.add_argument('--dry-run', action='store_true', help='Print commands only')
    parser.add_argument('--skip-existing', action='store_true', help='Skip if result exists')
    parser.add_argument('--parse-only', action='store_true', help='Only parse existing results')
    parser.add_argument('--booksim', default='./src/booksim', help='Path to booksim binary')
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                        help='Number of parallel processes (default: max CPU cores)')

    parser.add_argument('--schemes', nargs='+', type=str,
                        help='Specify which schemes to run (e.g., --schemes Baseline Fabric). Default: all',
                        default=None)
    parser.add_argument('--k-values', nargs='+', type=int,
                        help='Specify k values to run (e.g., --k-values 1 4). Default: all',
                        default=None)
    parser.add_argument('--sizes', nargs='+', type=int,
                        help='Specify sizes in MiB to run (e.g., --sizes 8 64). Default: all',
                        default=None)

    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt and start immediately after showing plan')

    # Routing comparison option
    parser.add_argument('--routings', nargs='+', type=str,
                        metavar='ROUTING',
                        help=(
                            'Specify one or more routing functions to use. '
                            'When multiple values are given, creates one run per routing '
                            'and prints a side-by-side comparison table. '
                            'When omitted, uses each scenario\'s default routing_function. '
                            f'Choices: {", ".join(ALL_ROUTING_FUNCTIONS)}'
                        ),
                        default=None)

    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)

    # Validate routing functions
    if args.routings:
        invalid = [r for r in args.routings if r not in ALL_ROUTING_FUNCTIONS]
        if invalid:
            print(f"Error: Unknown routing function(s): {invalid}")
            print(f"  Valid choices: {ALL_ROUTING_FUNCTIONS}")
            sys.exit(1)

    # Filter scenarios
    active_scenarios = SCENARIOS
    if args.schemes:
        wanted_schemes = [s.lower() for s in args.schemes]
        active_scenarios = [s for s in SCENARIOS if s["label"].lower() in wanted_schemes]
        if not active_scenarios:
            print(f"Error: None of the specified schemes {args.schemes} match available scenarios.")
            sys.exit(1)

    active_k_values = args.k_values if args.k_values else K_VALUES
    active_sizes    = args.sizes    if args.sizes    else SIZES_MIB

    # Determine routing variants
    # use_routing_suffix=True when multiple routings are given (to distinguish result filenames)
    if args.routings:
        active_routings     = args.routings
        use_routing_suffix  = len(args.routings) > 1
    else:
        active_routings    = None  # use each scenario's own routing_function
        use_routing_suffix = False

    csv_fields = [
        'job_num', 'scenario', 'routing_function', 'k', 'size_mib', 'total_mb',
        'total_packets', 'completion_time', 'time_taken',
        'moe_avg_pkt_latency', 'moe_avg_net_latency', 'moe_avg_hops',
        'pkt_lat_avg', 'pkt_lat_min', 'pkt_lat_max',
        'net_lat_avg',
        'flit_lat_avg', 'flit_lat_min', 'flit_lat_max',
        'fragmentation_avg',
        'inject_pkt_rate', 'accept_pkt_rate',
        'inject_flit_rate', 'accept_flit_rate',
        'hops_avg', 'run_time_sec',
        'ugal_nonmin_pct', 'ugal_total', 'ugal_min', 'ugal_nonmin',
        'escape_vc_pct', 'escape_vc_count', 'escape_vc_total',
        'matrix_file', 'result_file',
    ]
    csv_rows = []

    jobs = []
    job_num = 1
    for scenario in active_scenarios:
        routings_for_scenario = active_routings if active_routings else [scenario["routing_function"]]
        for routing in routings_for_scenario:
            for k in active_k_values:
                for size_mib in active_sizes:
                    jobs.append((scenario, routing, k, size_mib, args, use_routing_suffix, job_num, 0))
                    job_num += 1

    total_jobs = len(jobs)
    # Patch in correct total_jobs
    jobs = [(s, rf, k, sz, a, us, jn, total_jobs) for (s, rf, k, sz, a, us, jn, _) in jobs]

    succeeded, skipped, failed = 0, 0, 0

    # ── Plan summary ────────────────────────────────────────────
    print_plan_summary(active_scenarios, active_routings, active_k_values, active_sizes,
                       use_routing_suffix, RESULT_DIR, args)

    if not args.yes and not args.parse_only and not args.dry_run:
        try:
            input("Press Enter to start, or Ctrl-C to cancel... ")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
        print()

    rf_display = active_routings if active_routings else [s["routing_function"] for s in active_scenarios]
    print(f"Active schemes   : {[s['label'] for s in active_scenarios]}")
    print(f"Routing functions: {rf_display}")
    print(f"Active k values  : {active_k_values}")
    print(f"Active sizes     : {active_sizes}")
    print(f"Starting {total_jobs} jobs using {args.workers} parallel workers...\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_scenario, job): job for job in jobs}

        try:
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                print(res['msg'])

                status = res['status']
                if status == 'succeeded':
                    succeeded += 1
                elif status == 'failed':
                    failed += 1
                elif status in ['skipped', 'parsed']:
                    skipped += 1

                if res['row']:
                    csv_rows.append(res['row'])
        except KeyboardInterrupt:
            print("\n[WARNING] Keyboard Interrupt! Shutting down workers...")
            executor.shutdown(wait=False, cancel_futures=True)
            sys.exit(1)

    csv_rows.sort(key=lambda x: x.get('job_num', 0))

    csv_path = os.path.join(RESULT_DIR, "summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print()
    print("=" * 100)
    print(f"Total: {total_jobs} | Succeeded: {succeeded} | Failed: {failed} | Skipped: {skipped}")
    print(f"Summary CSV: {csv_path}")
    print("=" * 100)

    if csv_rows:
        print()
        hdr = (f"{'Scenario':<13} {'Routing':<15} {'k':>3} {'Size':>6} {'TotalMB':>10} "
               f"{'Packets':>10} {'Completion':>11} {'AvgLat':>9} {'Hops':>6} "
               f"{'UGAL NM%':>9} {'EscVC%':>7}")
        print(hdr)
        print("-" * len(hdr))
        for row in sorted(csv_rows, key=lambda x: (x.get('size_mib', 0), x.get('scenario', ''),
                                                    x.get('routing_function', ''), x.get('k', 0))):
            def _s(key, default=''):
                v = row.get(key, default)
                return '' if v is None else str(v)
            print(f"{_s('scenario'):<13} {_s('routing_function'):<15} {_s('k'):>3} "
                  f"{_s('size_mib')+'M':>6} "
                  f"{_s('total_mb'):>10} "
                  f"{_s('total_packets'):>10} "
                  f"{_s('completion_time'):>11} "
                  f"{_s('moe_avg_pkt_latency'):>9} "
                  f"{_s('moe_avg_hops'):>6} "
                  f"{_fmt_pct(row.get('ugal_nonmin_pct')):>9} "
                  f"{_fmt_pct(row.get('escape_vc_pct')):>7}")

        # Side-by-side comparison when multiple routings were run
        if active_routings and len(active_routings) > 1:
            print_routing_comparison(csv_rows, active_routings)


if __name__ == '__main__':
    main()
