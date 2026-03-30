#!/usr/bin/env python3
"""
Generate scaled (and optionally expanded) MoE traffic matrices for moe-GPU-HBM.

Row/column label types in this directory:
  Core{c}  : SM group attached to Crossbar c  (router-level, e.g. Core0 = SM 0..73)
  HBM{h}   : HBM router                       (router-level, e.g. HBM0 = L2 0..31)

Expanded label types (with --expand):
  SM{i}    : Individual SM   (fine-grained, expanded from Core)
  L2_{i}   : Individual L2 slice (fine-grained, expanded from HBM)

Expansion rules:
  Core{c} → SM{c*SMS_PER_CORE} .. SM{(c+1)*SMS_PER_CORE - 1}
  HBM{h}  → L2_{h*L2_PER_HBM} .. L2_{(h+1)*L2_PER_HBM - 1}
  Cell value ÷ (row_expansion_count × col_expansion_count)   (preserves total traffic)

Generated filenames:
  moe_matrix_{mode}_k{k}_{size}MiB           ← Core/HBM-level (always generated)
  moe_matrix_{mode}_k{k}_{size}MiB_sm        ← SM expanded rows/cols  (--expand sm|all)
  moe_matrix_{mode}_k{k}_{size}MiB_l2        ← L2 expanded rows/cols  (--expand l2|all)
  moe_matrix_{mode}_k{k}_{size}MiB_sm_l2     ← SM + L2 expanded       (--expand all)

Usage:
  python3 gen_all_matrices.py [--dir PATH] [--expand none|sm|l2|all] [--force]
                              [--num-sms 148] [--num-cores 2]
                              [--num-hbm 8]  [--num-l2 256]
"""

import os
import argparse

# ============================================================
#  Default configuration
# ============================================================
K_VALUES     = [1, 2, 4, 8, 16]
MODES        = ["baseline", "H2H"]
BASE_SIZE    = 128
TARGET_SIZES = [8, 16, 32, 64, 256]

DEFAULT_NUM_SMS   = 148
DEFAULT_NUM_CORES = 2
DEFAULT_NUM_HBM   = 8
DEFAULT_NUM_L2    = 256


# ============================================================
#  Label helpers
# ============================================================
def label_type(name):
    """Return 'Core', 'HBM', 'SM', 'L2', or None."""
    n = name.strip()
    if n.startswith('Core'): return 'Core'
    if n.startswith('HBM'):  return 'HBM'
    if n.startswith('SM'):   return 'SM'
    if n.startswith('L2_') or n == 'L2': return 'L2'
    return None

def is_data_label(name):
    return label_type(name) is not None

def expand_label(name, expand_sm, expand_l2, sms_per_core, l2_per_hbm):
    """Return list of expanded labels for a given label."""
    lt = label_type(name)
    if lt == 'Core' and expand_sm:
        c = int(name[4:])
        start = c * sms_per_core
        return [f"SM{j}" for j in range(start, start + sms_per_core)]
    if lt == 'HBM' and expand_l2:
        h = int(name[3:])
        start = h * l2_per_hbm
        return [f"L2_{j}" for j in range(start, start + l2_per_hbm)]
    return [name]


# ============================================================
#  Matrix I/O
# ============================================================
def parse_matrix(path):
    """
    Parse a tab-separated matrix file.

    Returns dict:
      'col_headers': list of column label strings
      'rows':        list of (row_label, [float values])
    """
    with open(path, 'r') as f:
        lines = [l.rstrip('\n\r') for l in f]

    col_headers = []
    rows = []

    for line in lines:
        parts = line.split('\t')
        if not any(p.strip() for p in parts):
            continue

        first = parts[0].strip()

        # Header row: first cell is not a data label and not "Col sum"
        if not is_data_label(first) and 'sum' not in first.lower():
            for p in parts[1:]:
                ps = p.strip()
                if not ps:
                    continue
                if 'sum' in ps.lower():
                    break
                if is_data_label(ps):
                    col_headers.append(ps)
            continue

        # Data row
        if is_data_label(first):
            vals = []
            for p in parts[1:]:
                ps = p.strip()
                if not ps:
                    continue
                try:
                    vals.append(float(ps))
                except ValueError:
                    break
            rows.append((first, vals[:len(col_headers)]))

        # Col sum row: skip (recomputed on write)

    return {'col_headers': col_headers, 'rows': rows}


def write_matrix(parsed, path):
    """Write a parsed matrix to file."""
    col_headers = parsed['col_headers']
    rows        = parsed['rows']

    with open(path, 'w') as f:
        f.write('\tDestination\n')
        f.write('Source\t' + '\t'.join(col_headers) + '\t\tRow sum\n')

        for label, vals in rows:
            row_sum = sum(vals)
            f.write(label + '\t' + '\t'.join(f'{v:.6f}' for v in vals)
                    + f'\t\t{row_sum:.6f}\n')

        if rows:
            n = len(col_headers)
            col_sums = [
                sum(rows[r][1][c] for r in range(len(rows)) if c < len(rows[r][1]))
                for c in range(n)
            ]
            total = sum(col_sums)
            f.write('Col sum\t' + '\t'.join(f'{v:.6f}' for v in col_sums)
                    + f'\t\t{total:.6f}\n')


def scale_matrix(parsed, ratio):
    """Return a new parsed matrix with all values multiplied by ratio."""
    return {
        'col_headers': list(parsed['col_headers']),
        'rows': [(lbl, [v * ratio for v in vals]) for lbl, vals in parsed['rows']],
    }


def expand_matrix(parsed, expand_sm, expand_l2, sms_per_core, l2_per_hbm):
    """
    Expand Core→SM and/or HBM→L2 labels.

    Values are divided by (row_expansion × col_expansion) to preserve total traffic.
    """
    old_cols = parsed['col_headers']

    # Build new columns and per-column expansion counts
    new_cols = []
    col_counts = []
    for c in old_cols:
        expanded = expand_label(c, expand_sm, expand_l2, sms_per_core, l2_per_hbm)
        col_counts.append(len(expanded))
        new_cols.extend(expanded)

    # Build new rows
    new_rows = []
    for lbl, vals in parsed['rows']:
        row_expanded = expand_label(lbl, expand_sm, expand_l2, sms_per_core, l2_per_hbm)
        row_count = len(row_expanded)

        new_vals = []
        for ci, cv in enumerate(vals):
            cc = col_counts[ci]
            cell_val = cv / (row_count * cc)
            new_vals.extend([cell_val] * cc)

        for new_lbl in row_expanded:
            new_rows.append((new_lbl, list(new_vals)))

    return {'col_headers': new_cols, 'rows': new_rows}


# ============================================================
#  Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate scaled/expanded MoE traffic matrices (moe-GPU-HBM).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dir', default=None,
                        help='Directory containing matrices (default: script location)')
    parser.add_argument('--expand', choices=['none', 'sm', 'l2', 'all'], default='none',
                        help=(
                            'Expansion granularity (default: none = Core/HBM-level only).\n'
                            '  sm  : Core → individual SMs\n'
                            '  l2  : HBM  → individual L2 slices\n'
                            '  all : both SM and L2 expansion'
                        ))
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing files')
    parser.add_argument('--num-sms',   type=int, default=DEFAULT_NUM_SMS,
                        help=f'Total SMs (default: {DEFAULT_NUM_SMS})')
    parser.add_argument('--num-cores', type=int, default=DEFAULT_NUM_CORES,
                        help=f'Number of Core groups (default: {DEFAULT_NUM_CORES})')
    parser.add_argument('--num-hbm',   type=int, default=DEFAULT_NUM_HBM,
                        help=f'Number of HBM stacks (default: {DEFAULT_NUM_HBM})')
    parser.add_argument('--num-l2',    type=int, default=DEFAULT_NUM_L2,
                        help=f'Total L2 slices (default: {DEFAULT_NUM_L2})')
    args = parser.parse_args()

    work_dir     = args.dir or os.path.dirname(os.path.abspath(__file__))
    sms_per_core = args.num_sms   // args.num_cores
    l2_per_hbm   = args.num_l2   // args.num_hbm
    expand_sm    = args.expand in ('sm',  'all')
    expand_l2    = args.expand in ('l2',  'all')

    print(f"Working directory : {work_dir}")
    print(f"K values          : {K_VALUES}")
    print(f"Modes             : {MODES}")
    print(f"Base size         : {BASE_SIZE} MiB")
    print(f"Target sizes      : {TARGET_SIZES} MiB")
    print(f"Expansion         : {args.expand}")
    if expand_sm:
        print(f"  Core → SM       : {args.num_cores} cores × {sms_per_core} SMs/core = {args.num_sms} SMs")
    if expand_l2:
        print(f"  HBM  → L2 slice : {args.num_hbm} HBMs  × {l2_per_hbm} L2/HBM  = {args.num_l2} L2 slices")
    print()

    # Determine which variant suffixes to generate
    variants = [('', False, False)]  # always generate base (no expansion)
    if expand_sm:
        variants.append(('_sm',    True,  False))
    if expand_l2:
        variants.append(('_l2',    False, True))
    if expand_sm and expand_l2:
        variants.append(('_sm_l2', True,  True))

    generated = skipped = missing = 0
    all_sizes  = sorted(set([BASE_SIZE] + TARGET_SIZES))
    last_base_ext = ''

    for k in K_VALUES:
        for mode in MODES:
            base_name = f"moe_matrix_{mode}_k{k}_{BASE_SIZE}MiB"
            base_path = None
            for ext in ['.txt', '']:
                c = os.path.join(work_dir, base_name + ext)
                if os.path.exists(c):
                    base_path = c
                    break

            if base_path is None:
                print(f"[SKIP] Base not found: {base_name}[.txt]")
                missing += 1
                continue

            print(f"[BASE] {os.path.basename(base_path)}")
            _, base_ext = os.path.splitext(base_path)
            last_base_ext = base_ext
            base_parsed = parse_matrix(base_path)

            for target in all_sizes:
                ratio   = target / BASE_SIZE
                scaled  = scale_matrix(base_parsed, ratio) if target != BASE_SIZE else base_parsed

                for suffix, do_sm, do_l2 in variants:
                    out_name = f"moe_matrix_{mode}_k{k}_{target}MiB{suffix}{base_ext}"
                    out_path = os.path.join(work_dir, out_name)

                    # Skip base file if it's the source itself and no suffix
                    if target == BASE_SIZE and suffix == '' and out_path == base_path:
                        continue

                    if not args.force and os.path.exists(out_path):
                        print(f"  [EXISTS] {out_name}")
                        skipped += 1
                        continue

                    if do_sm or do_l2:
                        data = expand_matrix(scaled, do_sm, do_l2, sms_per_core, l2_per_hbm)
                        n = len(data['col_headers'])
                        tag = f"({n}×{n})"
                    else:
                        data = scaled
                        tag = f"(×{ratio:.4f})" if target != BASE_SIZE else "(base copy)"

                    write_matrix(data, out_path)
                    print(f"  [GEN]    {out_name}  {tag}")
                    generated += 1

    print()
    print(f"Done: {generated} generated, {skipped} already existed, {missing} base files missing")

    # File existence table
    print()
    print("=== File existence check ===")
    for suffix, _, _ in variants:
        tag = f"({suffix.strip('_') or 'Core/HBM-level'})"
        header = f"  {'':35s}" + "".join(f"{s:>7d}" for s in all_sizes)
        print(f"\n  {tag}")
        print(header)
        for k in K_VALUES:
            for mode in MODES:
                label = f"{mode}_k{k}"
                row = f"  {label:35s}"
                for size in all_sizes:
                    name = f"moe_matrix_{mode}_k{k}_{size}MiB{suffix}"
                    found = any(
                        os.path.exists(os.path.join(work_dir, name + e))
                        for e in ([last_base_ext] if last_base_ext else ['', '.txt'])
                    )
                    row += f"{'✓':>7s}" if found else f"{'✗':>7s}"
                print(row)


if __name__ == '__main__':
    main()
