#!/usr/bin/env python3
"""
HBMNet Throughput Sweep Runner and Plotter.

Subcommands
-----------
  run   Run BookSim directly in throughput mode, sweeping injection rates.
  plot  Compare throughput sweep results in a single plot.

Examples
--------
  # Run: Sweep injection rate from 0.05 to 1.0 with step 0.05
  python3 run_throughput_sweep.py run \
      --structure B100_Global --bandwidth B100+HBM4e \
      --routing baseline near_min_adaptive \
      --near-min-k 1 2 \
      --initial-step 0.05 --minimum-step 0.05 --max-rate 1.0

  # Plot: Injection Rate vs Accepted Rate
  python3 run_throughput_sweep.py plot \
      --structure B100_Global --bandwidth B100+HBM4e \
      --routing baseline near_min_adaptive \
      --near-min-k 1 2 \
      --metric throughput

  # Plot: Accepted Rate vs Latency (Very common in NoC papers)
  python3 run_throughput_sweep.py plot \
      --structure B100_Global --bandwidth B100+HBM4e \
      --routing baseline near_min_adaptive \
      --near-min-k 1 2 \
      --metric acc_latency
"""

import argparse
import os
import re
import csv
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from experiments_loader import load_experiments

# ── Topology / bandwidth tables ───────────────────────────────────────────────

PORTS_PER_UNIT = 7
_HERE         = os.path.dirname(os.path.abspath(__file__))
STRUCTURES, BANDWIDTHS = load_experiments(os.path.join(_HERE, "experiments.csv"))
BASE_CONFIG   = os.path.join(_HERE, "src", "examples", "hbmnet_accelsim_config")
BOOKSIM_BIN   = os.path.join(_HERE, "src", "booksim")
RESULT_DIR    = os.path.join(_HERE, "results", "throughput_sweep")

ROUTING_CHOICES = [
    "baseline", "min_oblivious", "min_adaptive",
    "near_min_adaptive", "near_min_random", "fixed_min",
    "ugal", "valiant",
]

# ── Routing Helpers ───────────────────────────────────────────────────────────

def routing_to_key(routing: str, near_min_k: int = 2, near_min_p: float = 1.0) -> str:
    if routing == "near_min_adaptive":
        return f"near_min_adaptive_nmk{near_min_k}_nmp{near_min_p:.1f}"
    elif routing == "near_min_random":
        return f"near_min_random_nmk{near_min_k}"
    return routing

def routing_key_to_label(key: str) -> str:
    _LABELS = {
        "baseline":        "Baseline",
        "min_oblivious":   "Min-oblivious",
        "min_adaptive":    "Min-adaptive",
        "fixed_min":       "Fixed-min",
        "ugal":            "UGAL",
        "valiant":         "Valiant",
    }
    if key in _LABELS: return _LABELS[key]
    m_adp = re.match(r"near_min_adaptive_nmk(\d+)_nmp([\d.]+)$", key)
    if m_adp: return f"Near-min-adp (k={m_adp.group(1)}, p={m_adp.group(2)})"
    m_rnd = re.match(r"near_min_random_nmk(\d+)$", key)
    if m_rnd: return f"Near-min-rnd (k={m_rnd.group(1)})"
    return key

def routing_key_to_overrides(key: str, is_fabric: int) -> dict:
    ov = {"is_fabric": str(is_fabric), "routing_function": "hybrid"}
    if key == "baseline":
        ov["hybrid_routing"] = "baseline"
        return ov
    m_adp = re.match(r"^near_min_adaptive_nmk(\d+)_nmp([\d.]+)$", key)
    if m_adp:
        ov["hybrid_routing"], ov["near_min_k"], ov["near_min_penalty"] = "near_min_adaptive", m_adp.group(1), m_adp.group(2)
        return ov
    m_rnd = re.match(r"^near_min_random_nmk(\d+)$", key)
    if m_rnd:
        ov["hybrid_routing"], ov["near_min_k"] = "near_min_random", m_rnd.group(1)
        return ov
    ov["hybrid_routing"] = key
    return ov

# ── Config Generation ─────────────────────────────────────────────────────────

def generate_config(base_path: str, overrides: dict, output_path: str) -> None:
    with open(base_path) as f: lines = f.readlines()
    result = []
    written = set()
    for line in lines:
        s = line.strip()
        if s.startswith("//") or not s:
            result.append(line)
            continue
        m = re.match(r"^(\w+)\s*=", s)
        if m and m.group(1) in overrides:
            result.append(f"{m.group(1)} = {overrides[m.group(1)]};\n")
            written.add(m.group(1))
            continue
        result.append(line)
    for k, v in overrides.items():
        if k not in written: result.append(f"{k} = {v};\n")
    with open(output_path, "w") as f: f.writelines(result)

def struct_overrides(name: str) -> dict:
    s = STRUCTURES[name]
    return {"num_xbars": str(s["num_xbars"]), "hbm_per_side": str(s["hbm_per_side"]), "sm_per_xbar": str(s["sm_per_xbar"])}

def bw_overrides(name: str) -> dict:
    bw = BANDWIDTHS[name]
    p = PORTS_PER_UNIT
    return {
        "l2_per_hbm": str(bw["l2_per_hbm"]),
        "xbar_xbar_bandwidth": str(round(bw["gpu_gpu"] * p)),
        "xbar_hbm_bandwidth": str(round(bw["gpu_hbm"] * p)),
        "xbar_mc_bandwidth": str(round(bw["gpu_hbm"] * p)),
        "mc_hbm_bandwidth": str(round(bw["tsv"] * p)),
        "mc_mc_bandwidth": str(round(bw["hbm_hbm"] * p)),
    }

# ── BookSim Runner ────────────────────────────────────────────────────────────

def run_booksim_throughput(booksim: str, config_path: str) -> dict:
    proc = subprocess.run([booksim, config_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout
    if proc.returncode != 0: return {"Error": f"Failed with code {proc.returncode}"}
    
    lats = re.findall(r"Packet latency average = ([\d.]+)", out)
    accs = re.findall(r"Accepted packet rate average = ([\d.]+)", out)
    times = re.findall(r"Time taken is (\d+) cycles", out)
    
    if accs and lats and times:
        return {"Accepted_Rate": float(accs[-1]), "Latency": float(lats[-1]), "Time_Cycles": int(times[-1]), "Error": None}
    return {"Error": "Failed to parse metrics"}

def _run_sweep_worker(exp: dict, args, i: int, n: int) -> dict:
    struct, bw, rk = exp["structure"], exp["bandwidth"], exp["routing_key"]
    tag = f"[{i:2d}/{n}] {struct} / {bw} / {rk}"
    out_dir = os.path.join(args.result_dir, struct, bw, args.traffic_label, rk)
    out_csv = os.path.join(out_dir, "throughput_sweep.csv")

    if args.skip_existing and os.path.exists(out_csv):
        return {"status": "skipped", "msg": f"{tag} → [SKIP] Already exists."}

    overrides = {}
    overrides.update(struct_overrides(struct))
    overrides.update(bw_overrides(bw))
    overrides.update(routing_key_to_overrides(rk, args.is_fabric))
    overrides.update({
        "sim_type": "throughput",
        "baseline_ratio": str(args.baseline_ratio),
        "traffic": args.traffic,
        "injection_process": args.injection_process,
        "use_read_write": "1" if args.use_read_write else "0"
    })
    for kv in (args.override or []):
        k, _, v = kv.partition("=")
        overrides[k.strip()] = v.strip()

    if args.dry_run:
        return {"status": "dry-run", "msg": f"{tag} → [DRY RUN]"}

    os.makedirs(out_dir, exist_ok=True)
    results_list = []
    
    # Internal Sweep Loop
    curr_rate = args.initial_step
    while curr_rate <= args.max_rate + 1e-9:
        overrides["injection_rate"] = f"{curr_rate:.4f}"
        
        with tempfile.NamedTemporaryFile(suffix=".cfg", delete=False) as tmp:
            tmp_name = tmp.name
        
        try:
            generate_config(args.base_config, overrides, tmp_name)
            bs_res = run_booksim_throughput(args.booksim, tmp_name)
        finally:
            if os.path.exists(tmp_name): os.remove(tmp_name)

        if bs_res["Error"]:
            return {"status": "failed", "msg": f"{tag} → [ERROR at inj={curr_rate:.4f}] {bs_res['Error']}"}
        
        results_list.append({
            "Injection_Rate": curr_rate,
            "Accepted_Rate": bs_res["Accepted_Rate"],
            "Latency": bs_res["Latency"],
            "Time_Cycles": bs_res["Time_Cycles"]
        })
        curr_rate += args.minimum_step

    # Save to CSV
    keys = ["Injection_Rate", "Accepted_Rate", "Latency", "Time_Cycles"]
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results_list)

    return {"status": "ok", "msg": f"{tag} → [OK] Swept up to {args.max_rate:.4f}"}

def cmd_run(args):
    routing_keys = []
    for r in args.routing:
        if r == "near_min_adaptive":
            for k in args.near_min_k:
                for p in args.near_min_p: routing_keys.append(routing_to_key(r, k, p))
        elif r == "near_min_random":
            for k in args.near_min_k: routing_keys.append(routing_to_key(r, k))
        else: routing_keys.append(r)
    routing_keys = list(dict.fromkeys(routing_keys))

    experiments = [{"structure": s, "bandwidth": b, "routing_key": rk}
                   for s in args.structure for b in args.bandwidth for rk in routing_keys]
    n = len(experiments)

    print(f"Sweep Target : {args.initial_step} to {args.max_rate} (step: {args.minimum_step})")
    print(f"Experiments  : {n} (workers: {args.workers})\n")

    if args.dry_run: return

    succeeded, skipped, failed = 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_sweep_worker, exp, args, i, n): exp for i, exp in enumerate(experiments, 1)}
        for fut in as_completed(futures):
            res = fut.result()
            print(res["msg"])
            if res["status"] == "ok": succeeded += 1
            elif res["status"] == "skipped": skipped += 1
            else: failed += 1

    print(f"\nDone. Total: {n} | OK: {succeeded} | Skipped: {skipped} | Failed: {failed}")

# ── Plotter ───────────────────────────────────────────────────────────────────

METRIC_CONFIG = {
    "throughput":  {"x": "Injection_Rate", "y": "Accepted_Rate", "xlab": "Injection Rate (flits/node/cycle)", "ylab": "Accepted Rate", "title": "Throughput vs Injection Rate"},
    "latency":     {"x": "Injection_Rate", "y": "Latency",       "xlab": "Injection Rate (flits/node/cycle)", "ylab": "Latency (cycles)", "title": "Latency vs Injection Rate"},
    "acc_latency": {"x": "Accepted_Rate",  "y": "Latency",       "xlab": "Accepted Rate (flits/node/cycle)",  "ylab": "Latency (cycles)", "title": "Latency vs Accepted Rate"},
}

def parse_csv_file(filepath: str, x_col: str, y_col: str):
    xs, ys = [], []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row[x_col]))
            ys.append(float(row[y_col]))
    return np.array(xs), np.array(ys)

def _sample_at_grid(xs_raw, ys_raw, step: float):
    if len(xs_raw) == 0: return np.array([]), np.array([])
    # Sort strictly by X to allow interpolation
    sort_idx = np.argsort(xs_raw)
    xs_raw, ys_raw = xs_raw[sort_idx], ys_raw[sort_idx]
    xs_uniq, idx = np.unique(xs_raw, return_index=True)
    ys_uniq = ys_raw[idx]
    
    x_max = xs_uniq.max()
    tx = np.arange(step, x_max + step * 0.01, step)
    tx = tx[tx <= x_max]
    return tx, np.interp(tx, xs_uniq, ys_uniq)

def cmd_plot(args):
    mcfg = METRIC_CONFIG[args.metric]
    entries = []

    routing_keys = []
    for r in args.routing:
        if r == "near_min_adaptive":
            for k in args.near_min_k:
                for p in args.near_min_p: routing_keys.append(routing_to_key(r, k, p))
        elif r == "near_min_random":
            for k in args.near_min_k: routing_keys.append(routing_to_key(r, k))
        else: routing_keys.append(r)
    routing_keys = list(dict.fromkeys(routing_keys))

    structures, bandwidths = args.structure or list(STRUCTURES.keys()), args.bandwidth or list(BANDWIDTHS.keys())
    
    for struct in structures:
        for bw in bandwidths:
            for rk in routing_keys:
                csv_path = os.path.join(args.result_dir, struct, bw, args.traffic_label, rk, "throughput_sweep.csv")
                if not os.path.exists(csv_path):
                    print(f"[skip] not found: {csv_path}")
                    continue
                
                parts = []
                if len(structures) > 1: parts.append(struct)
                if len(bandwidths) > 1: parts.append(bw)
                parts.append(routing_key_to_label(rk))
                entries.append({"path": csv_path, "label": " / ".join(parts)})

    if not entries:
        print("No data to plot.")
        return

    cmap = plt.cm.tab20
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p", "H", "d", "1", "2", "3", "4", "8", "+"]
    linestyles = ["-", "--", "-.", ":"] * 5

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, entry in enumerate(entries):
        xs_raw, ys_raw = parse_csv_file(entry["path"], mcfg["x"], mcfg["y"])
        if len(xs_raw) == 0: continue
        
        # Grid step interpolation
        xs, ys = _sample_at_grid(xs_raw, ys_raw, step=args.grid_step)
        
        ax.plot(xs, ys, color=cmap(i % 20), linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)], markersize=5, linewidth=1.5, label=entry["label"])

    ax.set_xlabel(mcfg["xlab"], fontsize=12)
    ax.set_ylabel(mcfg["ylab"], fontsize=12)
    ax.set_title(mcfg["title"], fontsize=13)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, ncol=1 if len(entries) <= 10 else 2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max([np.max(parse_csv_file(e["path"], mcfg["x"], mcfg["y"])[0]) for e in entries]) * 1.05)
    
    if args.metric in ["latency", "acc_latency"]:
        ax.set_ylim(0, 500) # You can adjust this to your typical saturation cutoff

    plt.tight_layout()
    if not args.no_save:
        out = args.output or os.path.join(args.result_dir, f"comparison_{args.metric}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.show()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HBMNet Throughput Sweep Runner and Plotter")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # RUN
    p = sub.add_parser("run", help="Run throughput sweep")
    p.add_argument("--structure", nargs="+", default=["B100_Global"], choices=list(STRUCTURES.keys()))
    p.add_argument("--bandwidth", nargs="+", default=["B100+HBM3e"], choices=list(BANDWIDTHS.keys()))
    p.add_argument("--routing", nargs="+", default=["near_min_adaptive"], choices=ROUTING_CHOICES)
    p.add_argument("--near-min-k", nargs="+", type=int, default=[2])
    p.add_argument("--near-min-p", nargs="+", type=float, default=[1.0])
    
    p.add_argument("--initial-step", type=float, default=0.05, help="Start injection rate (default: 0.05)")
    p.add_argument("--minimum-step", type=float, default=0.05, help="Sweep step size (default: 0.05)")
    p.add_argument("--max-rate", type=float, default=1.0, help="Max injection rate (default: 1.0)")
    
    p.add_argument("--traffic", default="gpu")
    p.add_argument("--injection-process", default="gpu_bernoulli")
    p.add_argument("--traffic-label", default=None)
    p.add_argument("--use-read-write", action="store_true")
    p.add_argument("--is-fabric", type=int, default=1, choices=[0, 1])
    p.add_argument("--baseline-ratio", type=float, default=0.0)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--base-config", default=BASE_CONFIG)
    p.add_argument("--booksim", default=BOOKSIM_BIN)
    p.add_argument("--result-dir", default=RESULT_DIR)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    # PLOT
    q = sub.add_parser("plot", help="Plot sweep results")
    q.add_argument("--structure", nargs="+")
    q.add_argument("--bandwidth", nargs="+")
    q.add_argument("--routing", nargs="+", required=True)
    q.add_argument("--near-min-k", nargs="+", type=int, default=[2])
    q.add_argument("--near-min-p", nargs="+", type=float, default=[1.0])
    q.add_argument("--traffic-label", default="gpu")
    q.add_argument("--use-read-write", action="store_true")
    q.add_argument("--result-dir", default=RESULT_DIR)
    q.add_argument("--metric", default="throughput", choices=list(METRIC_CONFIG.keys()))
    q.add_argument("--grid-step", type=float, default=0.05)
    q.add_argument("--output", default=None)
    q.add_argument("--no-save", action="store_true")

    args = parser.parse_args()

    if getattr(args, 'traffic_label', None) is None or args.traffic_label == "gpu":
        if getattr(args, 'use_read_write', False):
            args.traffic_label = "gpu_rw"
        elif getattr(args, 'traffic_label', None) is None:
            args.traffic_label = args.traffic

    if args.cmd == "run": cmd_run(args)
    else: cmd_plot(args)

if __name__ == "__main__":
    main()