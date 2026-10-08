#!/usr/bin/env python3
"""
AKI phase-velocity pass 2: rescue ambiguous pairs with a reference curve.

Run this AFTER all SLURM array tasks (pick_dispersion.py) have finished.
It builds a median reference curve c_ref from the unambiguous pass-1 results,
then re-runs AKI on every pair that came back ambiguous or without a curve,
using c_ref to resolve the branch.

Usage:
    python step0_phase_pass2.py
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phase_velocity import (measure_phase_velocity_aki, save_dispersion,
                            build_reference_curve, trim_edge_outliers)
from pick_dispersion import read_h5_stack, get_half, mode_separate
import config


def load_pass1_results():
    """Load all slim AKI results from pass 1."""
    results = []
    for f in sorted(glob.glob(os.path.join(config.OUTPUT_CACHE, '*_aki.npz'))):
        d = np.load(f, allow_pickle=False)
        results.append({
            'pair': str(d.get('pair', '')),
            'ok': bool(d.get('ok', False)),
            'ambiguous': bool(d.get('ambiguous', False)),
            'f': np.asarray(d.get('f', []), dtype=float),
            'c': np.asarray(d.get('c', []), dtype=float),
        })
    return results


def phase_output_path(sta1, sta2):
    return os.path.join(config.OUTPUT_PHASE, f'disp_phase_{sta1}_{sta2}_td.dat')


def main():
    print('=== AKI PASS 2: Build reference curve and rescue ambiguous pairs ===')

    results = load_pass1_results()
    n_ok = sum(r['ok'] and not r['ambiguous'] for r in results)
    n_ambig = sum(r.get('ambiguous', False) for r in results)
    print(f'Pass 1 results: {len(results)} pairs, {n_ok} unique branch, {n_ambig} ambiguous')

    if n_ok == 0:
        print('No unambiguous pairs from pass 1. Cannot build reference curve.')
        print('Check fig_aki/ diagnostic plots and consider adjusting parameters.')
        return

    # Build reference curve
    c_ref = build_reference_curve(results, min_pairs=config.REF_MIN_PAIRS)
    if c_ref is None:
        print(f'Not enough pairs (need {config.REF_MIN_PAIRS}) to build c_ref.')
        return

    # Save reference curve
    ref_file = os.path.join(config.OUTPUT_ROOT, 'c_ref.dat')
    np.savetxt(ref_file, np.column_stack(c_ref),
               header='freq_Hz c_km_s', comments='# ')
    print(f'Reference curve: {c_ref[0].size} frequencies, '
          f'{c_ref[0].min():.2f}-{c_ref[0].max():.2f} Hz')
    print(f'Saved to {ref_file}')

    # Pass 2: re-run pairs without a phase curve
    n_rescued = 0
    cached = sorted(glob.glob(os.path.join(config.OUTPUT_CACHE, '*.npz')))
    # Filter to only pair caches (not _aki.npz)
    cached = [f for f in cached if not f.endswith('_aki.npz')]

    for npz_file in cached:
        d = np.load(npz_file, allow_pickle=False)
        sta1, sta2 = str(d['sta1']), str(d['sta2'])

        # Skip pairs that already have a phase curve from pass 1
        if os.path.exists(phase_output_path(sta1, sta2)):
            continue

        h5_file = str(d['h5_file'])
        if not os.path.exists(h5_file):
            continue

        distance = float(d['distance'])
        T_vg = d['T_vg']
        vg = d['vg']

        if T_vg.size < 2:
            continue

        # Read and fold G_LR0
        ccf = read_h5_stack(h5_file, config.STACK_TYPE,
                            components=['ZZ', 'RR', 'ZR', 'RZ'])
        if not all(c in ccf['data'] for c in ['ZZ', 'RR', 'ZR', 'RZ']):
            continue

        dt = ccf['dt']
        ZZ_s = get_half(ccf['data']['ZZ'], side='sym')
        RR_s = get_half(ccf['data']['RR'], side='sym')
        ZR_s = get_half(ccf['data']['ZR'], side='sym')
        RZ_s = get_half(ccf['data']['RZ'], side='sym')
        G_LR0, _ = mode_separate(ZZ_s, RR_s, RZ_s, ZR_s)

        aki_kw = dict(
            dist_km=distance,
            max_lag_s=config.AKI_LAG_FACTOR * distance / config.VG_MIN,
            c_min=config.AKI_C_MIN, c_max=config.AKI_C_MAX)

        res = measure_phase_velocity_aki(
            G_LR0, dt=dt, T_vg=T_vg, vg=vg,
            c_ref=c_ref, verbose=False, **aki_kw)

        if res['ok']:
            f_w, c_w, _ = trim_edge_outliers(res['f'], res['c'])
            if f_w.size >= 2:
                save_dispersion(phase_output_path(sta1, sta2),
                                1.0 / f_w, c_w, distance)
                n_rescued += 1

    print(f'Pass 2: {n_rescued} pairs rescued with c_ref '
          f'(out of {len(cached)} cached)')

    # Count total phase curves
    n_phase = len(glob.glob(os.path.join(config.OUTPUT_PHASE, '*.dat')))
    print(f'Total phase-velocity curves: {n_phase}')


if __name__ == '__main__':
    main()
