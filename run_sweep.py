#!/usr/bin/env python3
"""
HBMNet injection-rate sweep runner and plotter.

Subcommands
-----------
  run   Build configs, call sweep.sh for each experiment combination.
  plot  Compare sweep results in a single plot.

Directory layout
----------------
  {result_dir}/{structure}/{bandwidth}/{traffic_label}/{routing_key}/sweep.txt

Routing keys
------------
  baseline                                   routing_function=hybrid, hybrid_routing=baseline
  min_oblivious                              routing_function=hybrid, hybrid_routing=min_oblivious
  min_adaptive                               routing_function=hybrid, hybrid_routing=min_adaptive
  fixed_min                                  routing_function=hybrid, hybrid_routing=fixed_min
  near_min_adaptive_nmk<K>_nmp<P>            routing_function=hybrid, hybrid_routing=near_min_adaptive, k=K, p=P
  near_min_random_nmk<K>                     routing_function=hybrid, hybrid_routing=near_min_random, k=K
  ugal                                       routing_function=hybrid, hybrid_routing=ugal
  valiant                                    routing_function=hybrid, hybrid_routing=valiant

Examples
--------
  # Run baseline + near-min combinations for B100_Global
  python3 run_sweep.py run \
      --structure B100_Global \
      --bandwidth B100+HBM4e \
      --routing baseline near_min_adaptive near_min_random \
      --near-min-k 1 2 \
      --near-min-p 0.0 1.0 \
      --traffic gpu --injection-process gpu_bernoulli \
      --use-read-write

  # Plot: compare routings including different k and p values (with read/write traffic)
  python3 run_sweep.py plot \
      --structure B100_Global --bandwidth B100+HBM4e \
      --routing baseline min_adaptive near_min_adaptive near_min_random \
      --near-min-k 1 2 \
      --near-min-p 1.0 \
      --use-read-write
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from experiments_loader import load_experiments

# ── Topology / bandwidth tables loaded from experiments.csv ──────────────────

PORTS_PER_UNIT = 7

# Script-relative defaults
_HERE         = os.path.dirname(os.path.abspath(__file__))

STRUCTURES, BANDWIDTHS = load_experiments(
    os.path.join(_HERE, "experiments.csv")
)
BASE_CONFIG   = os.path.join(_HERE, "src", "examples", "hbmnet_accelsim_config")
BOOKSIM_BIN   = os.path.join(_HERE, "src", "booksim")
SWEEP_SH      = os.path.join(_HERE, "utils", "sweep.sh")
RESULT_DIR    = os.path.join(_HERE, "results", "sweep")

# ── Routing helpers ───────────────────────────────────────────────────────────

ROUTING_CHOICES = [
    "baseline", "min_oblivious", "min_adaptive",
    "near_min_adaptive", "near_min_random", "fixed_min",
    "ugal", "valiant",
]


def routing_to_key(routing: str, near_min_k: int = 2, near_min_p: float = 1.0) -> str:
    """Convert routing name (+ optional K and P values) to a directory key."""
    if routing == "near_min_adaptive":
        return f"near_min_adaptive_nmk{near_min_k}_nmp{near_min_p:.1f}"
    elif routing == "near_min_random":
        return f"near_min_random_nmk{near_min_k}"
    return routing


def routing_key_to_label(key: str) -> str:
    """Human-readable legend label for a routing key."""
    _LABELS = {
        "baseline":        "Baseline",
        "min_oblivious":   "Min-oblivious",
        "min_adaptive":    "Min-adaptive",
        "fixed_min":       "Fixed-min",
        "ugal":            "UGAL",
        "valiant":         "Valiant",
    }
    if key in _LABELS:
        return _LABELS[key]
    
    m_adp = re.match(r"near_min_adaptive_nmk(\d+)_nmp([\d.]+)$", key)
    if m_adp:
        return f"Near-min-adp (k={m_adp.group(1)}, p={m_adp.group(2)})"
    
    m_rnd = re.match(r"near_min_random_nmk(\d+)$", key)
    if m_rnd:
        return f"Near-min-rnd (k={m_rnd.group(1)})"
    
    return key


def routing_key_to_overrides(key: str, is_fabric: int) -> dict:
    """Config overrides for a given routing key."""
    ov = {"is_fabric": str(is_fabric), "routing_function": "hybrid"}
    if key == "baseline":
        ov["hybrid_routing"] = "baseline"
        return ov
    
    m_adp = re.match(r"^near_min_adaptive_nmk(\d+)_nmp([\d.]+)$", key)
    if m_adp:
        ov["hybrid_routing"]   = "near_min_adaptive"
        ov["near_min_k"]       = m_adp.group(1)
        ov["near_min_penalty"] = m_adp.group(2)
        return ov
    
    m_rnd = re.match(r"^near_min_random_nmk(\d+)$", key)
    if m_rnd:
        ov["hybrid_routing"]   = "near_min_random"
        ov["near_min_k"]       = m_rnd.group(1)
        return ov
    
    ov["hybrid_routing"] = key
    return ov

# ── Config generation ─────────────────────────────────────────────────────────

def generate_config(base_path: str, overrides: dict, output_path: str) -> None:
    """Write a new booksim config from base_path with key overrides applied."""
    with open(base_path) as f:
        lines = f.readlines()
    result = []
    written: set[str] = set()
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
        if k not in written:
            result.append(f"{k} = {v};\n")
    with open(output_path, "w") as f:
        f.writelines(result)

# ── Structure / bandwidth override helpers ────────────────────────────────────

def struct_overrides(name: str) -> dict:
    s = STRUCTURES[name]
    return {"num_xbars":    str(s["num_xbars"]),
            "hbm_per_side": str(s["hbm_per_side"]),
            "sm_per_xbar":  str(s["sm_per_xbar"])}


def bw_overrides(name: str) -> dict:
    bw = BANDWIDTHS[name]
    p = PORTS_PER_UNIT
    return {
        "l2_per_hbm":          str(bw["l2_per_hbm"]),
        "xbar_xbar_bandwidth": str(round(bw["gpu_gpu"] * p)),
        "xbar_hbm_bandwidth":  str(round(bw["gpu_hbm"] * p)),
        "xbar_mc_bandwidth":   str(round(bw["gpu_hbm"] * p)),
        "mc_hbm_bandwidth":    str(round(bw["tsv"]     * p)),
        "mc_mc_bandwidth":     str(round(bw["hbm_hbm"] * p)),
    }

# ── Sweep runner ──────────────────────────────────────────────────────────────

def run_sweep(booksim: str, sweep_sh: str, config_path: str,
              output_path: str, initial_step: str = "0.05",
              minimum_step: str = "0.0002", dry_run: bool = False) -> bool:
    """Execute sweep.sh and save combined stdout to output_path."""
    cmd = ["sh", sweep_sh, booksim, config_path]
    if dry_run:
        print(f"  [DRY] {' '.join(cmd)}")
        print(f"        → {output_path}")
        return True
    env = os.environ.copy()
    env["initial_step"] = initial_step
    env["minimum_step"] = minimum_step
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env,
    )
    with open(output_path, "w") as f:
        f.write(proc.stdout)
    return "SWEEP: Parameter sweep complete." in proc.stdout


# ── Parse sweep log (same logic as plot_sweep.py) ────────────────────────────

def parse_sweep_file(filepath: str) -> list:
    """
    Returns list of (injection_rate, latency, accepted_rate, time_taken)
    for each successful simulation run, sorted by injection_rate.
    """
    with open(filepath) as f:
        content = f.read()
    blocks = re.split(r"OVERRIDE Parameter: injection_rate=", content)[1:]
    rows = []
    for block in blocks:
        if "SWEEP: Simulation run succeeded." not in block:
            continue
        lines = block.split("\n")
        try:
            ir = float(lines[0].strip())
        except ValueError:
            continue
        lats  = re.findall(r"Packet latency average = ([\d.]+)", block)
        accs  = re.findall(r"Accepted packet rate average = ([\d.]+)", block)
        times = re.findall(r"Time taken is (\d+) cycles", block)
        if lats and accs and times:
            rows.append((ir, float(lats[-1]), float(accs[-1]), int(times[-1])))
    rows.sort(key=lambda x: x[0])
    seen: dict = {}
    for r in rows:
        seen[r[0]] = r[1:]
    return [(ir,) + vals for ir, vals in sorted(seen.items())]


def _sample_at_grid(xs_raw, ys_raw, step: float = 0.02):
    if len(xs_raw) == 0:
        return np.array([]), np.array([])
    x_max = xs_raw.max()
    tx = np.arange(step, x_max + step * 0.01, step)
    tx = tx[tx <= x_max]
    return tx, np.interp(tx, xs_raw, ys_raw)

# ── Metric config ─────────────────────────────────────────────────────────────

METRIC_CONFIG = {
    "latency":    {
        "col":   1,
        "ylabel": "Packet Latency Average (cycles)",
        "title":  "Latency vs Injection Rate",
        "ylim":  (0, 5000),
    },
    "throughput": {
        "col":   2,
        "ylabel": "Accepted Packet Rate (flits/node/cycle)",
        "title":  "Throughput vs Injection Rate",
        "ylim":  (0, None),
    },
    "time": {
        "col":   3,
        "ylabel": "Time Taken (cycles)",
        "title":  "Simulation Time vs Injection Rate",
        "ylim":  (0, None),
    },
}

# ── RUN subcommand ────────────────────────────────────────────────────────────

def _run_one(exp: dict, args, i: int, n: int) -> dict:
    """Run a single experiment. Returns a result dict with msg and status."""
    struct = exp["structure"]
    bw     = exp["bandwidth"]
    rk     = exp["routing_key"]
    tag    = f"[{i}/{n}] {struct} / {bw} / {args.traffic_label} / {rk}"

    out_path = os.path.join(
        args.result_dir, struct, bw, args.traffic_label, rk, "sweep.txt"
    )

    if args.skip_existing and os.path.exists(out_path):
        if "SWEEP: Parameter sweep complete." in open(out_path).read():
            return {"status": "skipped", "msg": f"{tag}  →  [SKIP]"}

    overrides: dict[str, str] = {}
    overrides.update(struct_overrides(struct))
    overrides.update(bw_overrides(bw))
    overrides.update(routing_key_to_overrides(rk, args.is_fabric))
    overrides["baseline_ratio"]    = str(args.baseline_ratio)
    overrides["sim_type"]          = "latency"
    overrides["traffic"]           = args.traffic
    overrides["injection_process"] = args.injection_process
    
    # Read/Write config injection
    overrides["use_read_write"]    = "1" if args.use_read_write else "0"

    for kv in (args.override or []):
        k, _, v = kv.partition("=")
        overrides[k.strip()] = v.strip()

    tmp = tempfile.NamedTemporaryFile(
        suffix=".cfg", delete=False,
        prefix=f"sweep_{struct}_{bw}_{rk}_",
    )
    tmp.close()
    try:
        generate_config(args.base_config, overrides, tmp.name)
        ok = run_sweep(
            args.booksim, args.sweep_sh, tmp.name, out_path,
            initial_step=args.initial_step,
            minimum_step=args.minimum_step,
            dry_run=args.dry_run,
        )
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

    sat = ""
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.startswith("SWEEP: Saturation throughput:"):
                sat = "  " + line.strip()
                break

    status = "ok" if ok else "incomplete"
    marker = "[OK]" if ok else "[INCOMPLETE]"
    return {"status": status, "msg": f"{tag}  →  {marker}{sat}\n    {out_path}"}


def cmd_run(args) -> None:
    # Expand routing keys dynamically based on combinations
    routing_keys: list[str] = []
    for r in args.routing:
        if r == "near_min_adaptive":
            for k in args.near_min_k:
                for p in args.near_min_p:
                    routing_keys.append(routing_to_key(r, k, p))
        elif r == "near_min_random":
            for k in args.near_min_k:
                # Random does not use 'p', avoid duplicate runs
                routing_keys.append(routing_to_key(r, k))
        else:
            routing_keys.append(r)

    # Remove duplicates from near_min_random expanding multiple 'p' values
    routing_keys = list(dict.fromkeys(routing_keys))

    experiments = [
        {"structure": s, "bandwidth": b, "routing_key": rk}
        for s in args.structure
        for b in args.bandwidth
        for rk in routing_keys
    ]

    n = len(experiments)

    # ── Plan summary ──────────────────────────────────────────────────────────
    print(f"Experiments to run: {n}  (workers: {args.workers})")
    print()
    for i, exp in enumerate(experiments, 1):
        out_path = os.path.join(
            args.result_dir, exp["structure"], exp["bandwidth"],
            args.traffic_label, exp["routing_key"], "sweep.txt"
        )
        print(f"  [{i:2d}/{n}] {exp['structure']} / {exp['bandwidth']} "
              f"/ {args.traffic_label} / {exp['routing_key']}")
        print(f"         → {out_path}")
    print()

    if args.dry_run:
        print("(dry-run mode — no files will be written)")
        return

    if not args.yes:
        try:
            input("Press Enter to start, or Ctrl-C to cancel... ")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
    print()

    # ── Launch workers ────────────────────────────────────────────────────────
    print(f"Starting {n} jobs with {args.workers} workers...\n")
    succeeded, skipped, failed = 0, 0, 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_one, exp, args, i, n): exp
            for i, exp in enumerate(experiments, 1)
        }
        try:
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    print(res["msg"])
                    if res["status"] == "ok":         succeeded += 1
                    elif res["status"] == "skipped":  skipped   += 1
                    else:                             failed    += 1
                except Exception as exc:
                    print(f"[ERROR] {futures[fut]}: {exc}")
                    failed += 1
        except KeyboardInterrupt:
            print("\n[WARNING] Interrupted! Shutting down workers...")
            pool.shutdown(wait=False, cancel_futures=True)
            sys.exit(1)

    print(f"\nDone. Total: {n} | OK: {succeeded} | Skipped: {skipped} | Incomplete: {failed}")

# ── PLOT subcommand ───────────────────────────────────────────────────────────

def cmd_plot(args) -> None:
    mcfg = METRIC_CONFIG[args.metric]
    col  = mcfg["col"]

    entries: list[dict] = []

    if args.paths:
        # Mode 1: direct paths
        for idx, p in enumerate(args.paths):
            sweep = p if p.endswith("sweep.txt") else os.path.join(p, "sweep.txt")
            if not os.path.exists(sweep):
                print(f"[skip] not found: {sweep}")
                continue
            if args.labels and idx < len(args.labels):
                label = args.labels[idx]
            else:
                parts = os.path.normpath(os.path.dirname(sweep)).split(os.sep)
                label = " / ".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            entries.append({"path": sweep, "label": label})

    else:
        # Mode 2: filter-based
        if not args.routing:
            print("Error: --routing is required when not using --paths")
            sys.exit(1)

        routing_keys: list[str] = []
        for r in args.routing:
            if r == "near_min_adaptive":
                for k in args.near_min_k:
                    for p in args.near_min_p:
                        routing_keys.append(routing_to_key(r, k, p))
            elif r == "near_min_random":
                for k in args.near_min_k:
                    routing_keys.append(routing_to_key(r, k))
            else:
                routing_keys.append(r)

        routing_keys = list(dict.fromkeys(routing_keys))

        structures = args.structure or list(STRUCTURES.keys())
        bandwidths = args.bandwidth or list(BANDWIDTHS.keys())
        tl         = args.traffic_label
        multi_s    = len(structures) > 1
        multi_b    = len(bandwidths) > 1

        for struct in structures:
            for bw in bandwidths:
                for rk in routing_keys:
                    sweep = os.path.join(
                        args.result_dir, struct, bw, tl, rk, "sweep.txt"
                    )
                    if not os.path.exists(sweep):
                        print(f"[skip] not found: {sweep}")
                        continue
                    parts = []
                    if multi_s:
                        parts.append(struct)
                    if multi_b:
                        parts.append(bw)
                    parts.append(routing_key_to_label(rk))
                    entries.append({"path": sweep, "label": " / ".join(parts)})

    if not entries:
        print("No data to plot.")
        return

    # Build plot (Supports up to 20 configs without duplicate styles)
    cmap        = plt.cm.tab20
    markers     = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<",
                   ">", "p", "H", "d", "1", "2", "3", "4", "8", "+"]
    linestyles  = ["-", "--", "-.", ":"] * 5

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, entry in enumerate(entries):
        rows = parse_sweep_file(entry["path"])
        if not rows:
            print(f"[skip] no valid data: {entry['path']}")
            continue
        xs_raw = np.array([r[0] for r in rows])
        ys_raw = np.array([r[col] for r in rows])
        xs, ys = _sample_at_grid(xs_raw, ys_raw)
        
        # Cycle through up to 20 unique marker/color combinations
        ax.plot(
            xs, ys,
            color=cmap(i % 20),
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markersize=5, linewidth=1.5,
            label=entry["label"],
        )

    ax.set_xlabel("Injection Rate (flits/node/cycle)", fontsize=12)
    ax.set_ylabel(mcfg["ylabel"], fontsize=12)
    ax.set_title(mcfg["title"], fontsize=13)
    
    # Legend max layout tweaks to fit many items
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, ncol=1 if len(entries) <= 10 else 2)
    
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    ax.set_xlim(0, 1)
    ylo, yhi = mcfg["ylim"]
    if yhi:
        ax.set_ylim(ylo, yhi)
    else:
        ax.set_ylim(bottom=ylo)
    plt.tight_layout()

    if not args.no_save:
        out = args.output
        if out is None:
            out = os.path.join(args.result_dir, f"comparison_{args.metric}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.show()

# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HBMNet sweep runner and plotter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples\n--------")[1] if "Examples" in __doc__ else "",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── run ──────────────────────────────────────────────────────────────────
    p = sub.add_parser("run", help="Run sweep.sh experiments")

    p.add_argument("--structure", nargs="+", default=["B100_Global"],
                   choices=list(STRUCTURES.keys()),
                   metavar="STRUCT", help="GPU architecture(s)")
    p.add_argument("--bandwidth", nargs="+", default=["B100+HBM3e"],
                   choices=list(BANDWIDTHS.keys()),
                   metavar="BW", help="Bandwidth config(s)")
    p.add_argument("--routing", nargs="+", default=["near_min_adaptive"],
                   choices=ROUTING_CHOICES,
                   help="Routing function(s)")
    p.add_argument("--near-min-k", nargs="+", type=int, default=[2],
                   metavar="K",
                   help="near_min_k extra-hop budget (default: 2)")
    p.add_argument("--near-min-p", nargs="+", type=float, default=[1.0],
                   metavar="P",
                   help="near_min_penalty multiplier values for near_min_adaptive only (default: 1.0)")
    p.add_argument("--traffic", default="gpu",
                   help="booksim traffic= value  (default: gpu)")
    p.add_argument("--injection-process", default="gpu_bernoulli",
                   help="booksim injection_process= value  (default: gpu_bernoulli)")
    p.add_argument("--traffic-label", default=None,
                   metavar="LABEL",
                   help="Directory label for traffic (default: same as --traffic, appends _rw if use-read-write)")
    p.add_argument("--use-read-write", action="store_true",
                   help="Enable use_read_write=1 for request/reply separated traffic sizes")
    p.add_argument("--is-fabric", type=int, default=1, choices=[0, 1],
                   help="is_fabric: 0=no MC-MC links, 1=enable fabric  (default: 1)")
    p.add_argument("--baseline-ratio", type=float, default=0.0,
                   help="baseline_ratio (L2 hit rate, default: 0.0)")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Extra config overrides, e.g. --override num_vcs=8 near_min_strict=1")
    p.add_argument("--base-config", default=BASE_CONFIG,
                   help=f"Base booksim config  (default: {BASE_CONFIG})")
    p.add_argument("--booksim", default=BOOKSIM_BIN,
                   help=f"booksim binary  (default: {BOOKSIM_BIN})")
    p.add_argument("--sweep-sh", default=SWEEP_SH,
                   help=f"sweep.sh path  (default: {SWEEP_SH})")
    p.add_argument("--result-dir", default=RESULT_DIR,
                   help=f"Output root  (default: {RESULT_DIR})")
    p.add_argument("--initial-step", default="0.05",
                   help="sweep.sh initial_step env var  (default: 0.05)")
    p.add_argument("--minimum-step", default="0.0002",
                   help="sweep.sh minimum_step env var  (default: 0.0002)")
    p.add_argument("--workers", type=int, default=os.cpu_count(), metavar="N",
                   help=f"Number of parallel workers (default: {os.cpu_count()})")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip experiments with a complete sweep.txt already present")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip the Enter confirmation prompt")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan without executing")

    # ── plot ─────────────────────────────────────────────────────────────────
    q = sub.add_parser("plot", help="Compare sweep results in one plot")

    # Filter-based selection
    q.add_argument("--structure", nargs="+", metavar="STRUCT",
                   help="Structure(s) to include (default: all available)")
    q.add_argument("--bandwidth", nargs="+", metavar="BW",
                   help="Bandwidth(s) to include (default: all available)")
    q.add_argument("--traffic-label", default="gpu", metavar="LABEL",
                   help="Traffic label directory (default: gpu or gpu_rw based on --use-read-write)")
    q.add_argument("--use-read-write", action="store_true",
                   help="Look for the _rw traffic-label suffix automatically")
    q.add_argument("--routing", nargs="+", metavar="KEY",
                   help="Routing(s) to plot: baseline min_adaptive near_min_adaptive near_min_random ugal valiant "
                        "(or already-expanded keys like near_min_adaptive_nmk2_nmp1.5)")
    q.add_argument("--near-min-k", nargs="+", type=int, default=[2], metavar="K",
                   help="near_min_k values when --routing includes near_min_adaptive/random (default: 2)")
    q.add_argument("--near-min-p", nargs="+", type=float, default=[1.0], metavar="P",
                   help="near_min_penalty values when --routing includes near_min_adaptive (default: 1.0)")

    # Direct-path mode
    q.add_argument("--paths", nargs="*", metavar="PATH",
                   help="Direct paths to sweep result directories")
    q.add_argument("--labels", nargs="*", metavar="LABEL",
                   help="Legend labels for --paths (one per path)")

    q.add_argument("--result-dir", default=RESULT_DIR,
                   help=f"Result root for filter mode  (default: {RESULT_DIR})")
    q.add_argument("--metric", default="latency",
                   choices=list(METRIC_CONFIG.keys()),
                   help="Y-axis metric  (default: latency)")
    q.add_argument("--output", default=None,
                   help="Output PNG path (auto-named if omitted)")
    q.add_argument("--no-save", action="store_true",
                   help="Show plot without saving PNG")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Automatically set traffic labels based on read/write status
    if args.cmd == "run" and getattr(args, 'traffic_label', None) is None:
        args.traffic_label = args.traffic
        if args.use_read_write:
            args.traffic_label += "_rw"
            
    if args.cmd == "plot" and getattr(args, 'traffic_label', "gpu") == "gpu":
        if getattr(args, 'use_read_write', False):
            args.traffic_label = "gpu_rw"

    if args.cmd == "run":
        cmd_run(args)
    else:
        cmd_plot(args)


if __name__ == "__main__":
    main()