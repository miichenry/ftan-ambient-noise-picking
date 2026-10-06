#!/usr/bin/env python3
"""
Merge all per-pair dispersion CSV files into one consolidated table.

Applies basic quality-control filters (SNR and distance/wavelength ratio)
and writes a single CSV file for histogram plotting and tomographic inversion.

Usage:
    python step1_merge_picks.py
"""
import os
import glob
import time
import pandas as pd
import config

t0 = time.time()

# Input: all per-pair CSV files
csv_pattern = os.path.join(config.OUTPUT_DISP, '*.csv')
csv_files = sorted(glob.glob(csv_pattern))
print(f'Found {len(csv_files)} CSV files in {config.OUTPUT_DISP}')

if len(csv_files) == 0:
    print('No CSV files found. Run pick_dispersion.py first.')
    raise SystemExit(1)

os.makedirs(config.OUTPUT_MERGED, exist_ok=True)

# Merge all files
dfs = []
for k, f in enumerate(csv_files):
    if (k + 1) % 500 == 0:
        print(f'  Reading {k + 1}/{len(csv_files)}...')
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f'  WARNING: Could not read {f}: {e}. Skipping.')
        continue

    # Extract station names from filename
    bname = os.path.basename(f).replace('_group_sym.csv', '').replace('_group_pos.csv', '').replace('_group_neg.csv', '')
    parts = bname.split('_')
    df['stasrc'] = parts[0] if len(parts) >= 1 else ''
    df['starcv'] = parts[1] if len(parts) >= 2 else ''

    # Apply QC
    df = df[(df['snr_nb'] >= config.MERGE_SNR_THRESH) &
            (df['ratio_d_lambda'] >= config.MERGE_DLAMBDA_THRESH)]

    if len(df) > 0:
        dfs.append(df)

if not dfs:
    print('No picks survived QC filtering.')
    raise SystemExit(1)

picks = pd.concat(dfs, ignore_index=True)
print(f'Total picks after QC: {len(picks)}')

# Write output
out_file = os.path.join(
    config.OUTPUT_MERGED,
    f'merged_picks_snr{config.MERGE_SNR_THRESH}_dlambda{config.MERGE_DLAMBDA_THRESH}.csv'
)
picks.to_csv(out_file, index=False)
print(f'Written to {out_file}')
print(f'Time: {time.time() - t0:.1f} s')
