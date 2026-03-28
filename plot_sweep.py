#!/usr/bin/env python3
"""Plot injection rate vs packet latency (or throughput) from booksim2 sweep results.

Usage:
    python plot_sweep.py --traffic sm-to-l2 --architecture B200
    python plot_sweep.py --traffic all-to-all --architecture Rubin_Ultra --metric throughput
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# (file suffix, display label) for each routing
ROUTINGS_B200 = [
    ('base',   'Baseline'),
    ('adp',    'Min-adaptive'),
    ('obv',    'Min-oblivious'),
    ('near00', 'Near-min-adaptive P=0.0'),
    ('near05', 'Near-min-adaptive P=0.5'),
    ('near10', 'Near-min-adaptive P=1.0'),
    ('near15', 'Near-min-adaptive P=1.5'),
]

ROUTINGS_RUBIN = [
    ('rubin-base',   'Baseline'),
    ('rubin-adp',    'Min-adaptive'),
    ('rubin-obv',    'Min-oblivious'),
    ('rubin-near00', 'Near-min-adaptive P=0.0'),
    ('rubin-near05', 'Near-min-adaptive P=0.5'),
    ('rubin-near10', 'Near-min-adaptive P=1.0'),
    ('rubin-near15', 'Near-min-adaptive P=1.5'),
]

TRAFFIC_TITLES = {
    'sm-to-l2':   'SM→L2',
    'all-to-all': 'All-to-All',
}

ARCH_TITLES = {
    'B200':        'B200',
    'Rubin_Ultra': 'Rubin Ultra',
}


def parse_sweep_file(filepath):
    """Parse a sweep result file.

    For each successful injection rate block, returns the LAST value of both
    'Packet latency average' and 'Accepted packet rate average'.

    Returns a list of (injection_rate, latency, accepted_rate) sorted by injection_rate.
    """
    with open(filepath) as f:
        content = f.read()

    blocks = re.split(r'OVERRIDE Parameter: injection_rate=', content)[1:]
    rows = []
    for block in blocks:
        if 'SWEEP: Simulation run succeeded.' not in block:
            continue
        lines = block.split('\n')
        try:
            ir = float(lines[0].strip())
        except ValueError:
            continue
        lats  = re.findall(r'Packet latency average = ([\d.]+)', block)
        accs  = re.findall(r'Accepted packet rate average = ([\d.]+)', block)
        times = re.findall(r'Time taken is (\d+) cycles', block)
        if lats and accs and times:
            rows.append((ir, float(lats[-1]), float(accs[-1]), int(times[-1])))

    rows.sort(key=lambda x: x[0])
    # Deduplicate by injection rate — keep last occurrence
    seen = {}
    for ir, lat, acc, t in rows:
        seen[ir] = (lat, acc, t)
    return [(ir, lat, acc, t) for ir, (lat, acc, t) in sorted(seen.items())]


def sample_at_grid(xs_raw, ys_raw, step=0.02):
    """Linearly interpolate y values at a uniform injection-rate grid."""
    if len(xs_raw) == 0:
        return np.array([]), np.array([])
    x_max = xs_raw.max()
    target_xs = np.arange(step, x_max + step * 0.01, step)
    target_xs = target_xs[target_xs <= x_max]
    target_ys = np.interp(target_xs, xs_raw, ys_raw)
    return target_xs, target_ys


METRIC_CONFIG = {
    'latency': {
        'extract': lambda rows: [(r[0], r[1]) for r in rows],
        'ylabel':  'Packet Latency Average (cycles)',
        'title':   'Latency vs Injection Rate',
        'ylim':    (0, 5000),
    },
    'throughput': {
        'extract': lambda rows: [(r[0], r[2]) for r in rows],
        'ylabel':  'Accepted Packet Rate (flits/node/cycle)',
        'title':   'Throughput vs Injection Rate',
        'ylim':    (0, None),
    },
    'time': {
        'extract': lambda rows: [(r[0], r[3]) for r in rows],
        'ylabel':  'Time Taken (cycles)',
        'title':   'Simulation Time vs Injection Rate',
        'ylim':    (0, None),
    },
}


def plot_architecture(traffic, architecture, results_dir, metric='latency', save_png=True):
    routings = ROUTINGS_B200 if architecture == 'B200' else ROUTINGS_RUBIN
    mcfg = METRIC_CONFIG[metric]

    cmap       = plt.cm.tab10
    colors     = [cmap(i / 10) for i in range(len(routings))]
    markers    = ['o', 's', '^', 'D', 'v', 'P', 'X']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.']

    fig, ax = plt.subplots(figsize=(10, 6))

    plotted = 0
    for i, (suffix, label) in enumerate(routings):
        filepath = os.path.join(results_dir, f'sweep-{suffix}.txt')
        if not os.path.exists(filepath):
            print(f'  [skip] {os.path.basename(filepath)} not found')
            continue

        triples = parse_sweep_file(filepath)
        if not triples:
            print(f'  [skip] {os.path.basename(filepath)} has no data')
            continue

        pairs = mcfg['extract'](triples)
        xs_raw = np.array([p[0] for p in pairs])
        ys_raw = np.array([p[1] for p in pairs])
        xs, ys = sample_at_grid(xs_raw, ys_raw, step=0.02)

        ax.plot(
            xs, ys,
            color=colors[i],
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markersize=5,
            linewidth=1.5,
            label=label,
        )
        plotted += 1

    if plotted == 0:
        print('No data found — nothing to plot.')
        plt.close(fig)
        return

    arch_title    = ARCH_TITLES[architecture]
    traffic_title = TRAFFIC_TITLES.get(traffic, traffic)

    ax.set_xlabel('Injection Rate (flits/node/cycle)', fontsize=12)
    ax.set_ylabel(mcfg['ylabel'], fontsize=12)
    ax.set_title(f'{arch_title} — {traffic_title}: {mcfg["title"]}', fontsize=13)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    ax.set_xlim(0, 1)
    ax.set_ylim(*mcfg['ylim'])

    plt.tight_layout()

    if save_png:
        out_path = os.path.join(results_dir, f'plot_{traffic}_{architecture}_{metric}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot booksim2 sweep latency-throughput curves'
    )
    parser.add_argument(
        '--traffic',
        choices=['sm-to-l2', 'all-to-all'],
        required=True,
        help='Traffic pattern (matches results/ subdirectory name)',
    )
    parser.add_argument(
        '--architecture',
        choices=['B200', 'Rubin_Ultra'],
        required=True,
        help='Architecture to plot',
    )
    parser.add_argument(
        '--metric',
        choices=['latency', 'throughput', 'time'],
        default='latency',
        help='Y-axis metric: latency (default) or throughput (accepted packet rate)',
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Show plot without saving PNG',
    )
    args = parser.parse_args()

    results_dir = os.path.join(RESULTS_BASE, args.traffic)
    if not os.path.isdir(results_dir):
        print(f'Error: directory not found: {results_dir}')
        return

    print(f'Plotting {ARCH_TITLES[args.architecture]} / {TRAFFIC_TITLES.get(args.traffic, args.traffic)} [{args.metric}]')
    plot_architecture(args.traffic, args.architecture, results_dir,
                      metric=args.metric, save_png=not args.no_save)


if __name__ == '__main__':
    main()
