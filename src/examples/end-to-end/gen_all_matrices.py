#!/usr/bin/env python3
"""
Generate scaled (and optionally expanded) MoE traffic matrices.

Row/column label types supported:
  HBM{h}   : HBM router (router-level)
  L2_{i}   : Individual L2 slice  (fine-grained, expanded from HBM)
  SM{i}    : Individual SM        (fine-grained, expanded from Core — not in this dir)

Scaling:
  Takes a base 128MiB matrix and generates scaled copies for other sizes.

Expansion (--expand):
  HBM{h} row/col  →  L2_{h*L2_PER_HBM} ... L2_{(h+1)*L2_PER_HBM-1}
  Cell value is divided by (row_expansion_count × col_expansion_count).
  Total traffic is preserved.

Generated filenames:
  moe_matrix_{mode}_k{k}_{size}MiB          ← HBM-level (default)
  moe_matrix_{mode}_k{k}_{size}MiB_l2       ← L2-slice expanded  (--expand l2)

Usage:
  python3 gen_all_matrices.py [--dir PATH] [--expand none|l2] [--force]
                              [--num-hbm 8] [--num-l2 256]
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

# Hardware defaults (override with CLI flags)
DEFAULT_NUM_HBM = 8
DEFAULT_NUM_L2  = 256   # total L2 slices → L2_PER_HBM = NUM_L2 / NUM_HBM


# ============================================================
#  Label helpers
# ============================================================
def label_type(name):
    """Return 'HBM', 'L2', 'SM', or None."""
    n = name.strip()
    if n.startswith('HBM'):  return 'HBM'
    if n.startswith('L2_') or n == 'L2': return 'L2'
    if n.startswith('SM'):   return 'SM'
    return None

def is_data_label(name):
    return label_type(name) is not None

def expand_label(name, expand_l2, l2_per_hbm):
    """Return list of expanded labels for a given source label."""
    lt = label_type(name)
    if lt == 'HBM' and expand_l2:
        h = int(name[3:])
        start = h * l2_per_hbm
        return [f"L2_{j}" for j in range(start, start + l2_per_hbm)]
    return [name]  # no expansion


# ============================================================
#  Matrix I/O
# ============================================================
def parse_matrix(path):
    """
    Parse a tab-separated matrix file.

    Returns dict:
      'col_headers': list of column label strings (excludes Row-sum column)
      'rows':        list of (row_label, [float values])  — data rows only
      'has_rowsum':  bool — whether file has a trailing Row sum column
    """
    with open(path, 'r') as f:
        lines = [l.rstrip('\n\r') for l in f]

    col_headers = []
    rows = []
    has_rowsum = False

    for line in lines:
        parts = line.split('\t')
        if not any(p.strip() for p in parts):
            continue  # blank line

        first = parts[0].strip()

        # Header row (first non-blank part is "Source" or "Source\\Dest" etc.)
        if not is_data_label(first) and 'sum' not in first.lower() and first not in ('', 'Col sum'):
            # Detect column headers: skip the first cell, collect label columns
            col_headers = []
            for p in parts[1:]:
                ps = p.strip()
                if not ps:
                    continue
                if 'sum' in ps.lower():
                    has_rowsum = True
                    break
                if is_data_label(ps):
                    col_headers.append(ps)
            continue

        # Data row
        if is_data_label(first):
            vals = []
            data_parts = parts[1:]
            for p in data_parts:
                ps = p.strip()
                if not ps:
                    continue
                try:
                    vals.append(float(ps))
                except ValueError:
                    break   # hit a non-numeric (e.g. row sum label or trailing)
            # Trim to len(col_headers) values (drop row-sum column if present)
            rows.append((first, vals[:len(col_headers)]))

        # Col sum row — skip (will be recomputed on write)

    return {'col_headers': col_headers, 'rows': rows, 'has_rowsum': has_rowsum}


def write_matrix(parsed, path):
    """Write a parsed matrix to file in the original tab-separated format."""
    col_headers = parsed['col_headers']
    rows        = parsed['rows']

    with open(path, 'w') as f:
        # Header rows
        f.write('\tDestination\n')
        f.write('Source\t' + '\t'.join(col_headers) + '\t\tRow sum\n')

        # Data rows
        for label, vals in rows:
            row_sum = sum(vals)
            f.write(label + '\t' + '\t'.join(f'{v:.6f}' for v in vals)
                    + f'\t\t{row_sum:.6f}\n')

        # Col sum row
        if rows:
            n = len(col_headers)
            col_sums = [sum(rows[r][1][c] for r in range(len(rows)) if c < len(rows[r][1]))
                        for c in range(n)]
            total = sum(col_sums)
            f.write('Col sum\t' + '\t'.join(f'{v:.6f}' for v in col_sums)
                    + f'\t\t{total:.6f}\n')


def scale_matrix(parsed, ratio):
    """Return a new parsed matrix with all values multiplied by ratio."""
    return {
        'col_headers': list(parsed['col_headers']),
        'rows': [(lbl, [v * ratio for v in vals]) for lbl, vals in parsed['rows']],
        'has_rowsum': parsed['has_rowsum'],
    }


def expand_matrix(parsed, expand_l2, l2_per_hbm):
    """
    Expand HBM labels to L2-slice labels.

    Each HBM{h} row is split into L2_PER_HBM rows (L2_{h*lph} .. L2_{(h+1)*lph-1}).
    Each HBM{h} column is split into L2_PER_HBM columns.
    Cell values are divided by (row_count × col_count) so total traffic is preserved.
    """
    old_cols = parsed['col_headers']

    # Build new column list and expansion map col_idx → [new col indices]
    new_cols = []
    col_expand_count = []
    for c in old_cols:
        expanded = expand_label(c, expand_l2, l2_per_hbm)
        col_expand_count.append(len(expanded))
        new_cols.extend(expanded)

    # Build new rows
    new_rows = []
    for lbl, vals in parsed['rows']:
        row_expanded = expand_label(lbl, expand_l2, l2_per_hbm)
        row_count = len(row_expanded)

        # Build the expanded value list for this row (expand columns)
        new_vals = []
        for ci, cv in enumerate(vals):
            cc = col_expand_count[ci]
            cell_val = cv / (row_count * cc)
            new_vals.extend([cell_val] * cc)

        for new_lbl in row_expanded:
            new_rows.append((new_lbl, list(new_vals)))

    return {'col_headers': new_cols, 'rows': new_rows, 'has_rowsum': True}


# ============================================================
#  Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate scaled (and optionally expanded) MoE traffic matrices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dir', default=None,
                        help='Directory containing matrices (default: script location)')
    parser.add_argument('--expand', choices=['none', 'l2'], default='none',
                        help='Expansion granularity: none=HBM-level only (default), '
                             'l2=also generate L2-slice expanded versions')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing files')
    parser.add_argument('--num-hbm', type=int, default=DEFAULT_NUM_HBM,
                        help=f'Number of HBM stacks (default: {DEFAULT_NUM_HBM})')
    parser.add_argument('--num-l2', type=int, default=DEFAULT_NUM_L2,
                        help=f'Total L2 slices (default: {DEFAULT_NUM_L2})')
    args = parser.parse_args()

    work_dir   = args.dir or os.path.dirname(os.path.abspath(__file__))
    l2_per_hbm = args.num_l2 // args.num_hbm
    expand_l2  = args.expand == 'l2'

    print(f"Working directory : {work_dir}")
    print(f"K values          : {K_VALUES}")
    print(f"Modes             : {MODES}")
    print(f"Base size         : {BASE_SIZE} MiB")
    print(f"Target sizes      : {TARGET_SIZES} MiB")
    print(f"Expansion         : {args.expand}")
    if expand_l2:
        print(f"  HBM → L2 slices : {args.num_hbm} HBMs × {l2_per_hbm} L2/HBM = {args.num_l2} L2 slices")
    print()

    generated = skipped = missing = 0
    all_sizes  = sorted(set([BASE_SIZE] + TARGET_SIZES))

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
            base_parsed = parse_matrix(base_path)

            for target in all_sizes:
                ratio    = target / BASE_SIZE
                _, ext   = os.path.splitext(base_path)
                out_name = f"moe_matrix_{mode}_k{k}_{target}MiB{ext}"
                out_path = os.path.join(work_dir, out_name)

                if target != BASE_SIZE:
                    if not args.force and os.path.exists(out_path):
                        print(f"  [EXISTS] {out_name}")
                        skipped += 1
                    else:
                        scaled = scale_matrix(base_parsed, ratio)
                        write_matrix(scaled, out_path)
                        print(f"  [GEN]    {out_name}  (×{ratio:.4f})")
                        generated += 1
                else:
                    scaled = base_parsed  # no scaling needed for base size

                # Expanded (L2-slice) version
                if expand_l2:
                    out_l2_name = f"moe_matrix_{mode}_k{k}_{target}MiB_l2{ext}"
                    out_l2_path = os.path.join(work_dir, out_l2_name)
                    if not args.force and os.path.exists(out_l2_path):
                        print(f"  [EXISTS] {out_l2_name}")
                        skipped += 1
                    else:
                        src = scale_matrix(base_parsed, ratio) if target != BASE_SIZE else base_parsed
                        expanded = expand_matrix(src, expand_l2=True, l2_per_hbm=l2_per_hbm)
                        write_matrix(expanded, out_l2_path)
                        num_labels = len(expanded['col_headers'])
                        print(f"  [GEN]    {out_l2_name}  ({num_labels}×{num_labels} L2 slices)")
                        generated += 1

    print()
    print(f"Done: {generated} generated, {skipped} already existed, {missing} base files missing")

    # File existence table
    print()
    print("=== File existence check ===")
    variants = [''] + (['_l2'] if expand_l2 else [])
    for variant in variants:
        tag = f"({variant.strip('_') or 'HBM-level'})"
        header = f"{'':35s}" + "".join(f"{s:>7d}" for s in all_sizes)
        print(f"\n  {tag}")
        print("  " + header)
        for k in K_VALUES:
            for mode in MODES:
                label = f"{mode}_k{k}"
                row = f"  {label:35s}"
                for size in all_sizes:
                    _, ext = os.path.splitext(base_path) if base_path else ('', '')
                    name = f"moe_matrix_{mode}_k{k}_{size}MiB{variant}{ext}"
                    found = any(os.path.exists(os.path.join(work_dir, name + e))
                                for e in ([ext] if ext else ['', '.txt']))
                    row += f"{'✓':>7s}" if found else f"{'✗':>7s}"
                print(row)


if __name__ == '__main__':
    main()
