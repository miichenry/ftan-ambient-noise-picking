#!/usr/bin/env python3
"""
Plot pick-density histograms for all components.

Creates a multi-panel figure showing 2D histograms (period vs group velocity)
for each component and the mode-separated Green's functions, plus the mean
dispersion curve with +/- 2 sigma.

Usage:
    python step2_plot_histograms.py [merged_csv_file]
"""
import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import config


def get_mean_curve(periods, velocities):
    """Mean and std of group velocity at each unique period."""
    periods_uniq = np.unique(periods)
    gv_mean = np.array([np.mean(velocities[periods == p]) for p in periods_uniq])
    gv_std = np.array([np.std(velocities[periods == p]) for p in periods_uniq])
    return periods_uniq, gv_mean, gv_std


def plot_component(picks_df, ax, bins, title, dmax=None):
    """Plot a 2D histogram of picks for one component."""
    if len(picks_df) == 0:
        ax.text(0.5, 0.5, 'No picks', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_title(title)
        return None

    per = picks_df['inst_period'].values
    gv = picks_df['group_velocity'].values

    heatmap, xedges, yedges = np.histogram2d(per, gv, bins=bins)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    kwargs = {'vmin': 0}
    if dmax is not None:
        kwargs['vmax'] = dmax

    im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='inferno',
                   aspect='auto', **kwargs)

    # Mean curve
    periods_uniq, gv_mean, gv_std = get_mean_curve(per, gv)
    ax.plot(periods_uniq, gv_mean, 'w--', lw=2, label='Mean')
    ax.plot(periods_uniq, gv_mean + 2 * gv_std, 'w:', lw=1)
    ax.plot(periods_uniq, gv_mean - 2 * gv_std, 'w:', lw=1)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Period (s)')
    ax.set_ylabel('Group velocity (km/s)')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    n_pairs = len(picks_df.groupby(['stasrc', 'starcv']))
    ax.text(0.02, 0.95, f'N={len(picks_df)}\n{n_pairs} pairs',
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    return im


def main():
    # Find merged CSV
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        pattern = os.path.join(config.OUTPUT_MERGED, 'merged_picks_*.csv')
        files = sorted(glob.glob(pattern))
        if not files:
            print(f'No merged CSV found in {config.OUTPUT_MERGED}. Run step1_merge_picks.py first.')
            sys.exit(1)
        csv_file = files[-1]

    print(f'Reading {csv_file}')
    picks = pd.read_csv(csv_file)
    print(f'Total picks: {len(picks)}')

    # Histogram bins
    bins_T = np.arange(config.TMIN, config.TMAX + 0.1, 0.1)
    bins_V = np.arange(config.VG_MIN, config.VG_MAX, 0.01)
    bins = [bins_T, bins_V]

    # Components to plot
    components = ['ZZ', 'RR', 'ZR', 'RZ', 'G_LR0', 'G_LR1', 'TT']
    components = [c for c in components if c in picks['component'].unique()]

    ncols = 2
    nrows = (len(components) + 1) // 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    if nrows == 1:
        axs = axs.reshape(1, -1)

    for idx, comp in enumerate(components):
        row, col = divmod(idx, ncols)
        ax = axs[row, col]
        picks_comp = picks[picks['component'] == comp]
        im = plot_component(picks_comp, ax, bins, comp)
        if im is not None:
            fig.colorbar(im, ax=ax, shrink=0.7, label='# picks')

    # Hide unused
    for idx in range(len(components), nrows * ncols):
        row, col = divmod(idx, ncols)
        axs[row, col].set_visible(False)

    fig.suptitle('Group-velocity pick density by component', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_fig = os.path.join(config.OUTPUT_MERGED, 'pick_histograms.png')
    fig.savefig(out_fig, dpi=200, bbox_inches='tight')
    print(f'Saved histogram figure: {out_fig}')
    plt.close(fig)

    # --- Additional: distance-binned histograms for ZZ ---
    if 'ZZ' in picks['component'].unique():
        zz = picks[picks['component'] == 'ZZ'].copy()
        dist_bins = np.percentile(zz['distance'], [0, 25, 50, 75, 100])

        fig2, axs2 = plt.subplots(2, 2, figsize=(14, 10))
        for idx in range(4):
            row, col = divmod(idx, 2)
            ax = axs2[row, col]
            d_lo, d_hi = dist_bins[idx], dist_bins[idx + 1]
            sub = zz[(zz['distance'] >= d_lo) & (zz['distance'] <= d_hi)]
            title = f'ZZ  d=[{d_lo:.1f}, {d_hi:.1f}] km'
            im = plot_component(sub, ax, bins, title)
            if im is not None:
                fig2.colorbar(im, ax=ax, shrink=0.7, label='# picks')

        fig2.suptitle('ZZ picks by distance quartile', fontsize=14, fontweight='bold')
        fig2.tight_layout(rect=[0, 0, 1, 0.96])
        out_fig2 = os.path.join(config.OUTPUT_MERGED, 'pick_histograms_ZZ_by_distance.png')
        fig2.savefig(out_fig2, dpi=200, bbox_inches='tight')
        print(f'Saved distance-binned histogram: {out_fig2}')
        plt.close(fig2)


if __name__ == '__main__':
    main()
