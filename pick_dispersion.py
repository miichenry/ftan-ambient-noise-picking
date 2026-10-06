#!/usr/bin/env python3
"""
Rayleigh-wave group-velocity dispersion picking using FTAN.

Processes one H5 file (one station pair) at a time. Designed to be called from
a SLURM array job. For each pair it:

  1. Reads the stacked cross-correlation from the H5 file
  2. Symmetrizes (folds) the correlation to boost SNR
  3. Runs FTAN on each requested component (ZZ, RR, ZR, RZ)
  4. Synthesizes mode-separated Green's functions G_LR0 / G_LR1
  5. Picks group-velocity dispersion curves with quality control
  6. Saves picks to CSV + a diagnostic FTAN figure per pair

Usage:
    python pick_dispersion.py <h5_file>

The H5 file path is typically provided by the SLURM wrapper script.
"""
import sys
import os
import warnings
import numpy as np
import h5py
from obspy import read_inventory
from obspy.geodetics import gps2dist_azimuth
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from FTAN_refactored import calc_ftan
from robust_dispersion_clean import remove_dispersion_jumps
import config

warnings.filterwarnings('ignore')


# ============================================================================
# H5 READER
# ============================================================================

def read_h5_stack(h5_file, stack_type, components=None):
    """Read stacked cross-correlations from a NoisePy H5 file.

    Returns
    -------
    dict with keys:
        'data': {comp: np.ndarray} for each component found
        'dt', 'dist', 'maxlag', 'azi', 'baz': float metadata
    """
    result = {'data': {}}
    with h5py.File(h5_file, 'r') as f:
        grp = f[f'AuxiliaryData/{stack_type}']
        avail = list(grp.keys())
        if components is None:
            components = avail

        for comp in components:
            if comp not in avail:
                continue
            ds = grp[comp]
            result['data'][comp] = ds[:]
            # Read metadata from the first component found
            if 'dt' not in result:
                attrs = dict(ds.attrs)
                result['dt'] = float(attrs.get('dt', 0.02))
                result['dist'] = float(attrs.get('dist', 0.0))
                result['maxlag'] = float(attrs.get('maxlag', 30.0))
                result['azi'] = float(attrs.get('azi', 0.0))
                result['baz'] = float(attrs.get('baz', 0.0))
                result['latS'] = float(attrs.get('latS', 0.0))
                result['lonS'] = float(attrs.get('lonS', 0.0))
                result['latR'] = float(attrs.get('latR', 0.0))
                result['lonR'] = float(attrs.get('lonR', 0.0))

    return result


# ============================================================================
# SYMMETRIZE
# ============================================================================

def symmetrize(data):
    """Fold a two-sided correlation about zero lag into a one-sided signal.

    The causal (+lag) and anti-causal (-lag) halves are summed to improve SNR.
    Zero-lag sample is counted once.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    m = n // 2
    out = data[m:].copy()
    out[1:] += data[:m][::-1]
    return out


def get_half(data, side='pos'):
    """Extract one side of a two-sided correlation."""
    n = len(data)
    m = n // 2
    if side == 'pos':
        return data[m:].copy()
    elif side == 'neg':
        return data[:m + 1][::-1].copy()
    else:
        return symmetrize(data)


# ============================================================================
# SNR
# ============================================================================

def calculate_snr_broadband(signal, dt, dist, vg_min=0.5, vg_max=4.0):
    """Broadband SNR (dB): signal window (expected arrival) vs. late coda."""
    t = np.arange(len(signal)) * dt
    t_min = dist / vg_max
    t_max = dist / vg_min
    sig_mask = (t >= t_min) & (t <= t_max)
    noise_mask = t > t_max * 2
    if sig_mask.sum() < 1 or noise_mask.sum() < 1:
        return 0.0
    sig_mean = np.nanmean(np.abs(signal[sig_mask]))
    noise_mean = np.nanmean(np.abs(signal[noise_mask])) + 1e-12
    return 20 * np.log10(sig_mean / noise_mean)


# ============================================================================
# PEAK PICKING
# ============================================================================

def get_peak_curve(ftan_img, T_plot, vg_grid, mask=None, threshold=0.3):
    """Extract group-velocity dispersion curve from an FTAN image.

    Peaks are ALWAYS picked on the full (unmasked) image so the picker finds
    the true amplitude maximum at each period.  The mask is only used to
    REJECT picks whose (T, vg) cell falls in the invalid region -- never to
    constrain the search, which would drag picks toward the mask boundary.
    """
    img = np.asarray(ftan_img, dtype=float)
    n_T, n_vg = img.shape
    KSIZE = 15

    # Smooth the FULL image (no masking during smoothing or peak search)
    smoothed = np.full_like(img, -np.inf)
    for i in range(n_T):
        smoothed[i] = uniform_filter1d(img[i], size=min(KSIZE, n_vg))

    peak_idx = np.array([np.argmax(row) for row in smoothed])
    peak_vg = vg_grid[peak_idx]
    peak_amp = np.array([img[i, peak_idx[i]] for i in range(n_T)])

    # Row-relative amplitude threshold (computed on the full image)
    with np.errstate(all='ignore'):
        row_max = np.nanmax(img, axis=1)
    rel_amp = peak_amp / np.where(row_max > 0, row_max, 1.0)
    has_sig = rel_amp > threshold

    # Post-pick rejection: drop picks that land in the masked (invalid) zone
    if mask is not None:
        for i in range(n_T):
            if has_sig[i] and mask[i, peak_idx[i]]:
                has_sig[i] = False

    return T_plot[has_sig], peak_vg[has_sig]


def wavelength_mask(T2d, vg2d, distance, factor=1.0):
    """Boolean mask of cells failing the far-field criterion."""
    wavelength = vg2d * T2d
    return (factor * wavelength) >= distance


def fix_picking_plateaus(x, y, window_size=41, threshold=0.4):
    """Remove large plateau jumps and interpolate."""
    import pandas as pd
    from scipy.interpolate import interp1d

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0:
        return y

    s = pd.Series(y)
    baseline = s.rolling(window=window_size, center=True, min_periods=1).median()
    baseline = baseline.bfill().ffill()
    outliers = np.abs(s - baseline) > threshold
    y_cleaned = np.copy(y)
    y_cleaned[outliers] = np.nan

    valid_idx = ~np.isnan(y_cleaned)
    x_valid = x[valid_idx]
    y_valid = y_cleaned[valid_idx]

    if len(x_valid) == 0:
        return y
    if len(x_valid) < 4:
        kind = 'linear' if len(x_valid) >= 2 else 'nearest'
    else:
        kind = 'cubic'

    f_interp = interp1d(x_valid, y_valid, kind=kind, fill_value="extrapolate")
    return f_interp(x)


def clean_group_curve(T_pk, vg_pk, distance, far_field=True):
    """Clean a picked group-velocity curve."""
    T = np.asarray(T_pk, dtype=float)
    V = np.asarray(vg_pk, dtype=float)
    if T.size == 0:
        return T, V

    if far_field:
        t_keep, v_keep = [], []
        keep_from_here = False
        for t_i, vg_i in zip(T, V):
            if not keep_from_here:
                if vg_i * t_i <= distance:
                    keep_from_here = True
                else:
                    continue
            t_keep.append(t_i)
            v_keep.append(vg_i)
        T, V = np.asarray(t_keep, dtype=float), np.asarray(v_keep, dtype=float)
        if T.size == 0:
            return T, V

    T, V = remove_dispersion_jumps(T, V)
    T, V = np.asarray(T, dtype=float), np.asarray(V, dtype=float)
    if T.size == 0:
        return T, V

    y_fixed = np.asarray(fix_picking_plateaus(T, V, window_size=51, threshold=0.4), dtype=float)
    keep = np.isfinite(y_fixed) & (y_fixed >= 0.1)
    return T[keep], V[keep]


# ============================================================================
# MODE SEPARATION (Nayak & Thurber 2020)
# ============================================================================

def mode_separate(ZZ, RR, RZ, ZR):
    """Synthesize mode-separated Green's functions G_LR0 and G_LR1.

    G_LR0 enhances the fundamental Rayleigh mode;
    G_LR1 enhances the first higher mode.
    """
    RZ_corrected = np.fft.rfft(RZ) * np.exp(1j * np.pi / 2)
    ZR_corrected = np.fft.rfft(ZR) * np.exp(-1j * np.pi / 2)
    G_LR0 = ZZ + RR + np.fft.irfft(RZ_corrected + ZR_corrected, n=len(ZZ))
    G_LR1 = ZZ + RR + np.fft.irfft(-RZ_corrected - ZR_corrected, n=len(ZZ))
    return G_LR0, G_LR1


# ============================================================================
# DIAGNOSTIC PLOTTING
# ============================================================================

def plot_ftan_diagnostic(ftan_results, pair_name, distance, output_path):
    """Create a multi-panel diagnostic figure showing FTAN images and picks.

    Each panel shows:
    - The FTAN image (period vs group velocity) with far-field masking
    - Raw picks as white crosses
    - Cleaned picks as a cyan line
    - SNR-filtered picks marked differently

    For G_LR0, the cleaned+far-field-filtered final curve is shown in red.
    """
    n_panels = len(ftan_results)
    ncols = 2
    nrows = (n_panels + 1) // 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows))
    if nrows == 1:
        axs = axs.reshape(1, -1)

    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad(color=(0.15, 0.15, 0.15, 1.0))

    for idx, (name, res) in enumerate(ftan_results.items()):
        row, col = divmod(idx, ncols)
        ax = axs[row, col]

        T2d = res['T2d']
        vg2d = res['vg2d']
        img = res['ftan_img']
        snr_db = res['snr_db']
        T_plot = res['T_plot']
        vg_grid = res['vg_grid']

        # Far-field mask
        Tg, Vg = np.meshgrid(T_plot, vg_grid, indexing='ij')
        invalid_wl = wavelength_mask(Tg, Vg, distance, config.FAR_FIELD_FACTOR)

        # SNR mask
        low_snr = np.isnan(snr_db) | (snr_db < config.SNR_THRESHOLD_DB)
        invalid_snr = np.broadcast_to(low_snr[:, None], img.shape)

        invalid = invalid_wl | invalid_snr

        # Show full FTAN image as dim background, then overlay valid region brighter
        cmap_bg = plt.get_cmap('bone').copy()
        ax.pcolormesh(T2d, vg2d, img.T, cmap=cmap_bg, vmin=0, vmax=1,
                      shading='flat', alpha=0.35, zorder=1)
        img_masked = np.ma.masked_where(invalid, img)
        ax.pcolormesh(T2d, vg2d, img_masked.T, cmap=cmap, vmin=0, vmax=1,
                      shading='flat', zorder=2)

        # Far-field boundary line: at each period, the minimum valid velocity
        # wavelength = vg * T <= distance  =>  vg <= distance / T
        T_line = T_plot[T_plot > 0]
        vg_boundary = distance / (config.FAR_FIELD_FACTOR * T_line)
        vg_boundary = np.clip(vg_boundary, config.VG_MIN, config.VG_MAX)
        ax.plot(T_line, vg_boundary, '--', color='yellow', lw=1.0, alpha=0.7,
                zorder=3, label='Far-field limit')

        # Raw picks (no mask)
        T_raw, vg_raw = get_peak_curve(img, T_plot, vg_grid, mask=None, threshold=config.PICK_THRESHOLD)
        ax.scatter(T_raw, vg_raw, marker='x', s=12, linewidths=0.6, color='white',
                   alpha=0.5, zorder=4, label='Raw picks')

        # Masked picks
        T_mk, vg_mk = get_peak_curve(img, T_plot, vg_grid, mask=invalid, threshold=config.PICK_THRESHOLD)
        ax.scatter(T_mk, vg_mk, marker='+', s=20, linewidths=0.8, color='cyan',
                   zorder=5, label='QC picks')

        # Final cleaned curve (for the primary component)
        if T_mk.size > 2:
            T_clean, vg_clean = clean_group_curve(T_mk, vg_mk, distance, far_field=True)
            if T_clean.size > 1:
                ax.plot(T_clean, vg_clean, '-', color='red', linewidth=1.5,
                        zorder=6, label='Final curve')

        ax.set_title(f'{name}  (d={distance:.2f} km)', fontsize=10)
        ax.set_xlabel('Period (s)', fontsize=9)
        ax.set_ylabel('Group velocity (km/s)', fontsize=9)
        ax.set_xscale('log')
        ax.set_ylim(config.VG_MIN, config.VG_MAX)
        ax.legend(fontsize=6, loc='upper right')
        ax.tick_params(labelsize=8)

    # Hide unused panels
    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axs[row, col].set_visible(False)

    fig.suptitle(f'{pair_name}   dist = {distance:.3f} km', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_path, f'ftan_{pair_name}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_pair(h5_file):
    """Process one station-pair H5 file: FTAN + pick + plot + CSV."""

    # Parse pair name from filename (e.g. "SS.19237_SS.24184.h5")
    basename = os.path.basename(h5_file)
    pair_name = basename.replace('.h5', '')
    parts = pair_name.split('_')
    sta1 = parts[0]  # e.g. "SS.19237"
    sta2 = parts[1]  # e.g. "SS.24184"

    print(f'Processing {pair_name} ...')

    # Read data
    ccf = read_h5_stack(h5_file, config.STACK_TYPE, components=config.COMPONENTS + ['TT'])
    if not ccf['data']:
        print(f'  No data found in {basename}; skipping.')
        return

    dt = ccf['dt']
    dist = ccf['dist']

    # If distance is 0 or missing, compute from coordinates
    if dist <= 0:
        dist_m, azi, baz = gps2dist_azimuth(
            ccf['latS'], ccf['lonS'], ccf['latR'], ccf['lonR'])
        dist = dist_m / 1000.0
        azi = azi
        baz = baz
    else:
        azi = ccf.get('azi', 0.0)
        baz = ccf.get('baz', 0.0)

    if dist < 0.01:
        print(f'  Distance ~0 for {pair_name}; skipping (autocorrelation?).')
        return

    # Ensure output dirs exist
    os.makedirs(config.OUTPUT_DISP, exist_ok=True)
    os.makedirs(config.OUTPUT_FIGS, exist_ok=True)

    # FTAN parameters for this pair
    ftan_params = dict(
        dist_km=dist, alpha=config.ALPHA, alpha2=config.ALPHA2,
        TMIN=config.TMIN, TMAX=config.TMAX,
        vg_min=config.VG_MIN, vg_max=config.VG_MAX,
        n_vg=config.N_VG, sigma_t=5.0,
    )

    # Grids for picking (must match what calc_ftan uses internally)
    vg_grid = np.linspace(config.VG_MIN, config.VG_MAX, config.N_VG)
    T_plot = np.logspace(np.log10(config.TMIN), np.log10(config.TMAX), config.N_PERIOD)[::-1]

    # Process each lag type
    for lag_type in config.LAG_TYPES:
        ftan_results = {}
        csv_rows = []

        # --- Individual components ---
        for comp in config.COMPONENTS:
            if comp not in ccf['data']:
                continue

            sig = get_half(ccf['data'][comp], side=lag_type)
            T2d, vg2d, img, snr_db = calc_ftan(sig, dt=dt, **ftan_params)

            ftan_results[comp] = {
                'T2d': T2d, 'vg2d': vg2d, 'ftan_img': img, 'snr_db': snr_db,
                'T_plot': T_plot, 'vg_grid': vg_grid,
            }

            # Pick dispersion curve
            Tg, Vg = np.meshgrid(T_plot, vg_grid, indexing='ij')
            invalid_wl = wavelength_mask(Tg, Vg, dist, config.FAR_FIELD_FACTOR)
            low_snr = np.isnan(snr_db) | (snr_db < config.SNR_THRESHOLD_DB)
            invalid_snr = np.broadcast_to(low_snr[:, None], img.shape)
            invalid = invalid_wl | invalid_snr

            T_pk, vg_pk = get_peak_curve(img, T_plot, vg_grid, mask=invalid,
                                          threshold=config.PICK_THRESHOLD)

            if T_pk.size > 0:
                T_clean, vg_clean = clean_group_curve(T_pk, vg_pk, dist, far_field=True)
            else:
                T_clean, vg_clean = np.array([]), np.array([])

            # SNR at each picked period
            snr_at_pick = np.interp(T_clean, T_plot[::-1], snr_db[::-1]) if T_clean.size > 0 else np.array([])

            for i in range(len(T_clean)):
                wavelength_i = T_clean[i] * vg_clean[i]
                ratio_d_lambda = dist / wavelength_i if wavelength_i > 0 else 0
                csv_rows.append({
                    'inst_period': T_clean[i],
                    'group_velocity': vg_clean[i],
                    'snr_nb': snr_at_pick[i] if i < len(snr_at_pick) else 0,
                    'ratio_d_lambda': ratio_d_lambda,
                    'azimuth': azi,
                    'backazimuth': baz,
                    'distance': dist,
                    'lag': lag_type,
                    'component': comp,
                    'pick_method': 'ftan_argmax',
                })

        # --- Mode-separated: G_LR0 (fundamental) ---
        has_all_4 = all(c in ccf['data'] for c in ['ZZ', 'RR', 'ZR', 'RZ'])
        if has_all_4:
            ZZ_s = get_half(ccf['data']['ZZ'], side=lag_type)
            RR_s = get_half(ccf['data']['RR'], side=lag_type)
            ZR_s = get_half(ccf['data']['ZR'], side=lag_type)
            RZ_s = get_half(ccf['data']['RZ'], side=lag_type)
            G_LR0, G_LR1 = mode_separate(ZZ_s, RR_s, RZ_s, ZR_s)

            for gname, gsig in [('G_LR0', G_LR0), ('G_LR1', G_LR1)]:
                T2d, vg2d, img, snr_db = calc_ftan(gsig, dt=dt, **ftan_params)
                ftan_results[gname] = {
                    'T2d': T2d, 'vg2d': vg2d, 'ftan_img': img, 'snr_db': snr_db,
                    'T_plot': T_plot, 'vg_grid': vg_grid,
                }

                Tg, Vg = np.meshgrid(T_plot, vg_grid, indexing='ij')
                invalid_wl = wavelength_mask(Tg, Vg, dist, config.FAR_FIELD_FACTOR)
                low_snr_g = np.isnan(snr_db) | (snr_db < config.SNR_THRESHOLD_DB)
                invalid_snr_g = np.broadcast_to(low_snr_g[:, None], img.shape)
                invalid_g = invalid_wl | invalid_snr_g

                T_pk, vg_pk = get_peak_curve(img, T_plot, vg_grid, mask=invalid_g,
                                              threshold=config.PICK_THRESHOLD)
                if T_pk.size > 0:
                    T_clean, vg_clean = clean_group_curve(T_pk, vg_pk, dist, far_field=True)
                else:
                    T_clean, vg_clean = np.array([]), np.array([])

                snr_at_pick = np.interp(T_clean, T_plot[::-1], snr_db[::-1]) if T_clean.size > 0 else np.array([])
                for i in range(len(T_clean)):
                    wavelength_i = T_clean[i] * vg_clean[i]
                    ratio_d_lambda = dist / wavelength_i if wavelength_i > 0 else 0
                    csv_rows.append({
                        'inst_period': T_clean[i],
                        'group_velocity': vg_clean[i],
                        'snr_nb': snr_at_pick[i] if i < len(snr_at_pick) else 0,
                        'ratio_d_lambda': ratio_d_lambda,
                        'azimuth': azi,
                        'backazimuth': baz,
                        'distance': dist,
                        'lag': lag_type,
                        'component': gname,
                        'pick_method': 'ftan_argmax',
                    })

        # --- TT component (Love wave) if available ---
        if 'TT' in ccf['data']:
            sig_tt = get_half(ccf['data']['TT'], side=lag_type)
            T2d, vg2d, img, snr_db = calc_ftan(sig_tt, dt=dt, **ftan_params)
            ftan_results['TT'] = {
                'T2d': T2d, 'vg2d': vg2d, 'ftan_img': img, 'snr_db': snr_db,
                'T_plot': T_plot, 'vg_grid': vg_grid,
            }

            Tg, Vg = np.meshgrid(T_plot, vg_grid, indexing='ij')
            invalid_wl = wavelength_mask(Tg, Vg, dist, config.FAR_FIELD_FACTOR)
            low_snr_tt = np.isnan(snr_db) | (snr_db < config.SNR_THRESHOLD_DB)
            invalid_snr_tt = np.broadcast_to(low_snr_tt[:, None], img.shape)
            invalid_tt = invalid_wl | invalid_snr_tt

            T_pk, vg_pk = get_peak_curve(img, T_plot, vg_grid, mask=invalid_tt,
                                          threshold=config.PICK_THRESHOLD)
            if T_pk.size > 0:
                T_clean, vg_clean = clean_group_curve(T_pk, vg_pk, dist, far_field=True)
            else:
                T_clean, vg_clean = np.array([]), np.array([])

            snr_at_pick = np.interp(T_clean, T_plot[::-1], snr_db[::-1]) if T_clean.size > 0 else np.array([])
            for i in range(len(T_clean)):
                wavelength_i = T_clean[i] * vg_clean[i]
                ratio_d_lambda = dist / wavelength_i if wavelength_i > 0 else 0
                csv_rows.append({
                    'inst_period': T_clean[i],
                    'group_velocity': vg_clean[i],
                    'snr_nb': snr_at_pick[i] if i < len(snr_at_pick) else 0,
                    'ratio_d_lambda': ratio_d_lambda,
                    'azimuth': azi,
                    'backazimuth': baz,
                    'distance': dist,
                    'lag': lag_type,
                    'component': 'TT',
                    'pick_method': 'ftan_argmax',
                })

        # Write CSV
        if csv_rows:
            csv_file = os.path.join(config.OUTPUT_DISP, f'{pair_name}_group_{lag_type}.csv')
            with open(csv_file, 'w') as fout:
                header = 'inst_period,group_velocity,snr_nb,ratio_d_lambda,azimuth,backazimuth,distance,lag,component,pick_method\n'
                fout.write(header)
                for r in csv_rows:
                    fout.write(f"{r['inst_period']:.4f},{r['group_velocity']:.4f},"
                               f"{r['snr_nb']:.2f},{r['ratio_d_lambda']:.2f},"
                               f"{r['azimuth']:.2f},{r['backazimuth']:.2f},"
                               f"{r['distance']:.4f},{r['lag']},{r['component']},"
                               f"{r['pick_method']}\n")
            print(f'  Wrote {len(csv_rows)} picks to {os.path.basename(csv_file)}')
        else:
            print(f'  No picks for {pair_name} (lag={lag_type})')

        # Diagnostic figure
        if ftan_results:
            plot_ftan_diagnostic(ftan_results, f'{pair_name}_{lag_type}', dist, config.OUTPUT_FIGS)
            print(f'  Saved FTAN figure for {pair_name}_{lag_type}')


# ============================================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <h5_file>")
        sys.exit(1)

    h5_file = sys.argv[1]
    if not os.path.exists(h5_file):
        print(f"File not found: {h5_file}")
        sys.exit(1)

    process_pair(h5_file)
