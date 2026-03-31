#!/usr/bin/env python3
"""
Plot per-direction link utilization and avg saturation as 3D bar charts.

Layout: rows = bandwidths, columns = scheme_routing combinations.
Each cell is one 3D bar chart. Separate figures for 'util' and 'sat' metrics
when --metric=both.

Usage:
  # Compare routings matching run_all_moe.py syntax
  python3 plot_link_stats.py --scheme Offloading --structure B100_Global \
      --bandwidth Sanity_B100 \
      --routing baseline fixed_min min_oblivious near_min_random min_adaptive near_min_adaptive \
      --near-min-k 1 2 --near-min-p 0.0 0.2 0.5 1.0 \
      --k-value 2 16 --size 256 --injection-rate 1.0 \
      --matrix-type End-to-End \
      --output linkstats

  # Save to file
  python3 plot_link_stats.py ... --output linkstats.png
"""

import os
import re
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from experiments_loader import load_experiments

# ============================================================
#  Constants loaded from experiments.csv
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
STRUCTURES, _BANDWIDTHS_DICT = load_experiments(os.path.join(_HERE, "experiments.csv"))
BANDWIDTHS = list(_BANDWIDTHS_DICT.keys())

# Base result directory (updated to user's specification)
DEFAULT_RESULT_DIR = "./results/moe"


# ============================================================
#  Parsing
# ============================================================
def parse_result_file(filepath):
    """Parse a BookSim result file and extract link stats."""
    with open(filepath, 'r') as f:
        text = f.read()

    result = {}

    # Link Type Utilization
    link_util = {}
    for m in re.finditer(
            r'^\s+(XBAR_XBAR|XBAR_HBM|XBAR_MC|MC_HBM|MC_MC):\s*(\d+)\s*\(([\d.]+)%\)',
            text, re.MULTILINE):
        link_util[m.group(1)] = {'count': int(m.group(2)), 'pct': float(m.group(3))}
    m = re.search(r'Total link traversals:\s*(\d+)', text)
    result['link_util'] = link_util
    result['total_traversals'] = int(m.group(1)) if m else 0

    # Miss link avg saturation (aggregate)
    m = re.search(
        r'Miss link avg saturation:\s*XBAR_XBAR=([\d.]+)%\s*XBAR_MC=([\d.]+)%\s*MC_MC=([\d.]+)%',
        text)
    if m:
        result['agg_sat'] = {
            'XBAR_XBAR': float(m.group(1)),
            'XBAR_MC': float(m.group(2)),
            'MC_MC': float(m.group(3)),
        }
    else:
        result['agg_sat'] = {'XBAR_XBAR': 0, 'XBAR_MC': 0, 'MC_MC': 0}

    # Per-direction entries
    dir_entries = []
    for m in re.finditer(
            r'^\s{4}(\S+(?:\([^)]*\))?)\s*->\s*(\S+(?:\([^)]*\))?):\s*count=(\d+)\s*'
            r'\(([\d.]+)%\)\s*avg_sat=([\d.]+)%',
            text, re.MULTILINE):
        dir_entries.append({
            'src': m.group(1), 'dst': m.group(2),
            'count': int(m.group(3)), 'pct': float(m.group(4)),
            'avg_sat': float(m.group(5)),
        })
    result['dir_entries'] = dir_entries

    # Scalar metrics
    m = re.search(r'Time taken is\s*(\d+)\s*cycles', text)
    result['time_taken'] = int(m.group(1)) if m else None
    m = re.search(r'Packet latency average\s*=\s*([\d.]+)', text)
    result['pkt_lat_avg'] = float(m.group(1)) if m else None
    m = re.search(r'Hops average\s*=\s*([\d.]+)', text)
    result['hops_avg'] = float(m.group(1)) if m else None
    m = re.search(r'near-min ratio\s*=\s*([\d.]+)%', text)
    result['nearmin_pct'] = float(m.group(1)) if m else None
    m = re.search(r'Near-min path usage:\s*(\d+)\s*/\s*(\d+)\s*\(\s*([\d.]+)%\)', text)
    result['nearmin_path_pct'] = float(m.group(3)) if m else None
    m = re.search(r'\+2 hop decisions:\s*(\d+)', text)
    result['nearmin_plus2'] = int(m.group(1)) if m else None

    return result


def scheme_routing_dirname(scenario, routing, nm_k, nm_p):
    """Build directory name for scheme + routing combination including near-min params."""
    if scenario == "Baseline" or routing == "baseline":
        return f"{scenario}_baseline"
    base = f"{scenario}_{routing}"
    
    if routing == "near_min_adaptive":
        base += f"_nmk{nm_k}_nmp{nm_p}"
    elif routing == "near_min_random":
        base += f"_nmk{nm_k}"
        
    return base


def get_result_path(base_result_dir, struct_name, bw_name, sr_dirname, k, size_mib, ir=1.0):
    """Build result file path, handling injection rate naming."""
    if ir == 1.0:
        filename = f"k{k}_{size_mib}MiB.txt"
    else:
        filename = f"k{k}_{size_mib}MiB_ir{ir}.txt"
    return os.path.join(base_result_dir, struct_name, bw_name, sr_dirname, filename)


def canonical_router_name(name):
    """Strip coordinate suffix: MC0(c0r0) -> MC0"""
    m = re.match(r'^(Xbar\d+|MC\d+)', name)
    return m.group(1) if m else name


def build_router_list(dir_entries_list):
    """Collect all unique Xbar/MC router names, ordered."""
    names = set()
    for entries in dir_entries_list:
        for e in entries:
            names.add(canonical_router_name(e['src']))
            names.add(canonical_router_name(e['dst']))
    xbars = sorted([n for n in names if n.startswith('Xbar')],
                   key=lambda x: int(re.search(r'\d+', x).group()))
    mcs = sorted([n for n in names if n.startswith('MC')],
                 key=lambda x: int(re.search(r'\d+', x).group()))
    return xbars + mcs


# ============================================================
#  Summary text
# ============================================================
def make_summary_text(parsed, mode='util'):
    """Build compact summary for annotation box.
    mode='util' : show link utilization summary only
    mode='sat'  : show avg saturation summary only
    """
    lines = []

    if mode == 'util':
        lu = parsed['link_util']
        parts = []
        for lt in ['XBAR_XBAR', 'XBAR_MC', 'MC_MC']:
            if lt in lu and lu[lt]['count'] > 0:
                short = {'XBAR_XBAR': 'X-X', 'XBAR_MC': 'X-M', 'MC_MC': 'M-M'}[lt]
                parts.append(f"{short}={lu[lt]['pct']:.1f}%")
        if parts:
            lines.append("Util: " + " ".join(parts))
    else:  # sat
        agg = parsed['agg_sat']
        lines.append(f"Sat: X-X={agg['XBAR_XBAR']:.1f}% X-M={agg['XBAR_MC']:.1f}% M-M={agg['MC_MC']:.1f}%")

    # Near-min stats (if available)
    nm = []
    if parsed.get('nearmin_pct') is not None:
        nm.append(f"NM={parsed['nearmin_pct']:.1f}%")
    if parsed.get('nearmin_plus2') is not None and parsed['nearmin_plus2'] > 0:
        nm.append(f"+2={parsed['nearmin_plus2']:,}")
    if nm:
        lines.append(" ".join(nm))

    # Perf (always shown)
    perf = []
    if parsed.get('time_taken'):
        perf.append(f"Cycles={parsed['time_taken']:,}")
    if parsed.get('hops_avg'):
        perf.append(f"Hop={parsed['hops_avg']:.2f}")
    if perf:
        lines.append(" ".join(perf))

    return "\n".join(lines)


# ============================================================
#  3D bar plot
# ============================================================
def plot_3d_bars(ax, dir_entries, router_list, value_key, vmax, cmap, norm):
    """Draw 3D bar chart on given axes. Returns nothing (caller sets titles)."""
    n = len(router_list)
    dst_list = list(reversed(router_list))  # Dst axis: MC7..MC0, Xbar1, Xbar0
    src_to_idx = {name: i for i, name in enumerate(router_list)}
    dst_to_idx = {name: i for i, name in enumerate(dst_list)}

    grid = np.full((n, n), np.nan)
    for e in dir_entries:
        src = canonical_router_name(e['src'])
        dst = canonical_router_name(e['dst'])
        if src in src_to_idx and dst in dst_to_idx:
            grid[src_to_idx[src], dst_to_idx[dst]] = e[value_key]

    for si in range(n):
        for di in range(n):
            val = grid[si, di]
            if np.isnan(val):
                continue
            c = cmap(norm(val))
            ax.bar3d(si, di, 0, 0.6, 0.6, val,
                     color=c, alpha=0.88, edgecolor='k', linewidth=0.2)

    ax.set_xticks(np.arange(n) + 0.3)
    ax.set_xticklabels(router_list, rotation=35, ha='right', fontsize=6.5, fontweight='bold')
    ax.set_yticks(np.arange(n) + 0.3)
    ax.set_yticklabels(dst_list, rotation=-20, ha='left', fontsize=6.5, fontweight='bold')
    ax.set_xlabel('Src', fontsize=7, fontweight='bold', labelpad=-2)
    ax.set_ylabel('Dst', fontsize=7, fontweight='bold', labelpad=-2)
    ax.set_zlabel('%', fontsize=7, fontweight='bold', labelpad=0)
    ax.set_zlim(0, vmax * 1.1 if vmax > 0 else 1)
    ax.tick_params(axis='z', labelsize=5.5)
    ax.tick_params(axis='x', pad=-7)
    ax.tick_params(axis='y', pad=-7)
    ax.view_init(elev=28, azim=-55)


# ============================================================
#  Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Plot per-direction link stats as 3D bar chart grid",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Note: Using singular names to perfectly match run_all_moe.py arguments
    parser.add_argument('--structure', nargs='+', default=list(STRUCTURES.keys()))
    parser.add_argument('--bandwidth', nargs='+', default=BANDWIDTHS)
    parser.add_argument('--scheme', nargs='+', default=['Baseline', 'Fabric', 'Offloading'])
    parser.add_argument('--routing', nargs='+',
                        default=['baseline', 'min_adaptive', 'fixed_min',
                                 'near_min_adaptive', 'near_min_random'])
    parser.add_argument('--k-value', nargs='+', type=int, default=[16])
    parser.add_argument('--size', nargs='+', type=int, default=[256])
    
    parser.add_argument('--near-min-k', nargs='+', type=int, default=[2],
                        help='Near-min routing budget k (default: 2)')
    parser.add_argument('--near-min-p', nargs='+', type=float, default=[1.0],
                        help='Near-min routing penalty multiplier (default: 1.0)')
    parser.add_argument('--injection-rate', nargs='+', type=float, default=[1.0],
                        help='Injection rate for MoE traffic (default: 1.0)')
    
    parser.add_argument('--matrix-type', choices=['GPU-HBM', 'End-to-End'], default='GPU-HBM',
                        help='Traffic matrix type (determines subfolder in result dir)')
    parser.add_argument('--result-dir', default=DEFAULT_RESULT_DIR,
                        help='Base result directory (default: ./results/moe)')
    
    parser.add_argument('--output', '-o', default=None,
                        help='Save plot to file (png/pdf/svg). '
                             'With --metric=both, _util and _sat suffixes are added.')
    parser.add_argument('--metric', choices=['both', 'util', 'sat'], default='both',
                        help='Which metric: util, sat, or both (separate figures)')
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()

    # Base result directory includes matrix type: ./results/moe/GPU-HBM
    base_result_dir = os.path.join(args.result_dir, args.matrix_type)

    # --- Expand routing keys based on combinations ---
    # Rows = bandwidths, Columns = scheme_routing combos (column order preserved)
    col_keys = []  # List of tuples: (scheme, routing, nm_k, nm_p)
    seen_cols = set()
    
    for scheme in args.scheme:
        routings = ['baseline'] if scheme.lower() == 'baseline' else args.routing
        for r in routings:
            if r == "near_min_adaptive":
                for k in args.near_min_k:
                    for p in args.near_min_p:
                        key = (scheme, r, k, p)
                        if key not in seen_cols:
                            col_keys.append(key)
                            seen_cols.add(key)
            elif r == "near_min_random":
                for k in args.near_min_k:
                    key = (scheme, r, k, "")
                    if key not in seen_cols:
                        col_keys.append(key)
                        seen_cols.add(key)
            else:
                key = (scheme, r, "", "")
                if key not in seen_cols:
                    col_keys.append(key)
                    seen_cols.add(key)

    # --- Iterate over independent variables to create separate figures ---
    for struct in args.structure:
        for k_val in args.k_value:
            for size in args.size:
                for ir in args.injection_rate:
                    row_keys = []  # bw names that have at least one result
                    grid = {}      # (bw, col_idx) -> (label, parsed)

                    for bw in args.bandwidth:
                        has_any = False
                        for ci, (scheme, routing, nm_k, nm_p) in enumerate(col_keys):
                            sr_dir = scheme_routing_dirname(scheme, routing, nm_k, nm_p)
                            fpath = get_result_path(base_result_dir, struct, bw,
                                                    sr_dir, k_val, size, ir)
                            if not os.path.exists(fpath):
                                continue
                            try:
                                parsed = parse_result_file(fpath)
                            except Exception as e:
                                print(f"  [WARN] {fpath}: {e}", file=sys.stderr)
                                continue
                            
                            sr_label = f"{scheme}_{routing}"
                            if routing == "near_min_adaptive":
                                sr_label += f"\n(k={nm_k}, p={nm_p})"
                            elif routing == "near_min_random":
                                sr_label += f"\n(k={nm_k})"
                                
                            grid[(bw, ci)] = (sr_label, parsed)
                            has_any = True
                        if has_any:
                            row_keys.append(bw)

                    if not grid:
                        continue # No valid results for this struct/k/size/ir combination

                    # Global router list
                    all_dirs = [p['dir_entries'] for _, p in grid.values()]
                    router_list = build_router_list(all_dirs)

                    n_rows = len(row_keys)
                    n_cols = len(col_keys)

                    # Metrics to plot
                    metric_list = []
                    if args.metric in ('both', 'util'):
                        metric_list.append(('pct', 'Link Usage (%)'))
                    if args.metric in ('both', 'sat'):
                        metric_list.append(('avg_sat', 'Avg Saturation (%)'))

                    for value_key, metric_label in metric_list:
                        # Global z-range for this metric
                        vmax = 1
                        for _, parsed in grid.values():
                            for e in parsed['dir_entries']:
                                if e[value_key] > vmax:
                                    vmax = e[value_key]

                        cmap = matplotlib.colormaps['coolwarm']
                        norm = colors.Normalize(vmin=0, vmax=vmax)

                        # Figure size
                        cell_w = max(4.5, 3.5 + 0.25 * len(router_list))
                        cell_h = max(4.5, 3.5 + 0.25 * len(router_list))
                        fig_w = cell_w * n_cols + 1.8
                        fig_h = cell_h * n_rows + 1.5

                        fig = plt.figure(figsize=(fig_w, fig_h))
                        
                        ir_str = f"IR={ir}" if ir != 1.0 else ""
                        title_text = f"{metric_label}  —  {struct} ({args.matrix_type})  k{k_val}  {size}MiB  {ir_str}"
                        fig.suptitle(title_text, fontsize=15, fontweight='bold', y=0.995)

                        for ri, bw in enumerate(row_keys):
                            for ci, (scheme, routing, nm_k, nm_p) in enumerate(col_keys):
                                cell = grid.get((bw, ci))
                                idx = ri * n_cols + ci + 1
                                ax = fig.add_subplot(n_rows, n_cols, idx, projection='3d')
                                
                                if cell is None:
                                    ax.set_visible(False)
                                    continue

                                sr_label, parsed = cell
                                plot_3d_bars(ax, parsed['dir_entries'],
                                             router_list, value_key, vmax,
                                             cmap, norm)

                                # Title: column label on top row, row label on left col
                                title = f"{bw}\n{sr_label}"
                                ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

                                # Summary box
                                summary_mode = 'util' if value_key == 'pct' else 'sat'
                                summary = make_summary_text(parsed, mode=summary_mode)
                                ax.text2D(0.5, 0.94, summary,
                                          transform=ax.transAxes,
                                          fontsize=8, ha='center', va='top',
                                          family='monospace',
                                          bbox=dict(boxstyle='round,pad=0.3',
                                                    facecolor='lightyellow',
                                                    edgecolor='gray', alpha=0.9))

                        # Colorbar
                        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
                        sm.set_array([])
                        cbar_ax = fig.add_axes([0.95, 0.15, 0.015, 0.7])
                        cbar = fig.colorbar(sm, cax=cbar_ax)
                        cbar.set_label(metric_label, fontsize=12)
                        cbar.ax.tick_params(labelsize=10)

                        plt.subplots_adjust(left=0.02, right=0.93, top=0.93,
                                            bottom=0.02, wspace=0.08, hspace=0.25)

                        # Save or show
                        if args.output:
                            base, ext = os.path.splitext(args.output)
                            if not ext:
                                ext = '.png'
                            suffix = ''
                            if args.metric == 'both':
                                suffix = '_util' if value_key == 'pct' else '_sat'
                            
                            suffix += f"_{struct}_{args.matrix_type}_k{k_val}_{size}M"
                            if ir != 1.0:
                                suffix += f"_ir{ir}"
                                
                            outpath = f"{base}{suffix}{ext}"
                            fig.savefig(outpath, dpi=args.dpi, bbox_inches='tight')
                            print(f"Saved: {outpath}")
                            plt.close(fig)
                        else:
                            plt.show()
                            plt.close(fig)


if __name__ == '__main__':
    main()