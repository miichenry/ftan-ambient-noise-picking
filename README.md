# FTAN Rayleigh-Wave Dispersion Picking for Ambient Noise Tomography

Automated group-velocity dispersion measurement from ambient-noise cross-correlations using Frequency-Time Analysis (FTAN). Designed for dense seismic arrays and HPC (SLURM) batch processing.

## Method

This workflow implements a two-stage FTAN following Levshin et al. (1989, 1992) and the multi-component Rayleigh-wave mode-separation of Nayak & Thurber (2020):

1. **Basic FTAN** -- A bank of narrowband Gaussian filters (log-spaced in period) is applied to the symmetrized cross-correlation. For each filter, the Hilbert envelope is mapped from lag time to group velocity (`vg = dist/t`), producing a period-vs-group-velocity image whose bright ridge is the dispersion curve.

2. **Mode separation** -- The four inter-station components (ZZ, RR, ZR, RZ) are combined with +/- 90-degree phase shifts to synthesize Green's-function combinations G_LR0 (fundamental Rayleigh mode) and G_LR1 (first higher mode).

3. **Quality control** -- Picks are filtered by:
   - Global FTAN amplitude threshold (rejects noise-dominated periods)
   - Far-field wavelength criterion (`distance >= factor * wavelength`)
   - Per-period narrowband SNR
   - Automatic mode-jump removal and plateau detection

4. **Diagnostic plots** -- Each station pair produces a multi-panel figure showing the full FTAN image, the valid region, the far-field boundary, raw picks, QC-filtered picks, and the final cleaned dispersion curve.

## File Overview

| File | Description |
|------|-------------|
| `config.py` | All paths and parameters -- **edit this first** |
| `FTAN_refactored.py` | Core FTAN engine (Gaussian filter bank, envelope, group-velocity mapping) |
| `robust_dispersion_clean.py` | Automatic mode-jump detection and removal |
| `pick_dispersion.py` | Main per-pair processing: read H5, FTAN, pick, plot, write CSV |
| `create_filelist.sh` | Generate the list of H5 files for the SLURM array job |
| `run_picking.slurm` | SLURM array job script |
| `step1_merge_picks.py` | Merge all per-pair CSVs into one table with QC filtering |
| `step2_plot_histograms.py` | Plot pick-density histograms per component |

## Quick Start

### 1. Configure

Edit `config.py` to set your project paths, period range, velocity range, and quality thresholds:

```python
# Key parameters to adjust:
CC_PATH = '/path/to/your/stacked/h5/files'
STATIONXML = '/path/to/stations.xml'
TMIN = 0.2      # shortest period (s)
TMAX = 5.0      # longest period (s) -- set by max inter-station distance
VG_MIN = 0.5    # group velocity min (km/s)
VG_MAX = 4.0    # group velocity max (km/s)
```

**Period range guideline**: Set `TMAX` so that `VG_MAX * TMAX` does not exceed your maximum inter-station distance. For a 2.5 km array, `TMAX ~ 3-5 s` is appropriate.

### 2. Generate file list

```bash
bash create_filelist.sh
```

This creates `h5_filelist.txt` with all H5 paths and prints the suggested `--array` range for SLURM.

### 3. Submit the picking job

Update the `--array` range in `run_picking.slurm` based on the output of step 2, then:

```bash
sbatch run_picking.slurm
```

Each array task processes ~100 station pairs. Output goes to:
- `dispersion_csv/` -- one CSV per pair with all picked periods, velocities, SNR, etc.
- `ftan_figures/` -- one diagnostic PNG per pair

### 4. Merge picks

After all SLURM tasks finish:

```bash
python step1_merge_picks.py
```

Produces a single consolidated CSV in `merged/` with SNR and distance/wavelength QC applied.

### 5. Plot histograms

```bash
python step2_plot_histograms.py
```

Produces:
- `pick_histograms.png` -- 2D pick-density histogram per component with mean +/- 2sigma
- `pick_histograms_ZZ_by_distance.png` -- ZZ picks split by distance quartile

## Input Data Format

Expects NoisePy-style stacked cross-correlation H5 files with structure:

```
AuxiliaryData/
  Allstack_linear/    (or Allstack_pws, configurable via STACK_TYPE)
    ZZ  (shape: 2*maxlag/dt + 1)
    RR
    ZR
    RZ
    TT  (optional, for Love waves)
    ...
```

Each dataset has attributes: `dt`, `dist`, `maxlag`, `azi`, `baz`, `latS`, `lonS`, `latR`, `lonR`.

## Output CSV Format

Each per-pair CSV contains:

| Column | Description |
|--------|-------------|
| `inst_period` | Period (s) |
| `group_velocity` | Group velocity (km/s) |
| `snr_nb` | Narrowband SNR at this period (dB) |
| `ratio_d_lambda` | Distance / wavelength ratio |
| `azimuth` | Source-to-receiver azimuth (degrees) |
| `backazimuth` | Back-azimuth (degrees) |
| `distance` | Inter-station distance (km) |
| `lag` | Lag type: sym, pos, or neg |
| `component` | Component: ZZ, RR, ZR, RZ, G_LR0, G_LR1, TT |
| `pick_method` | Always `ftan_argmax` |

## Diagnostic Figures

Each pair produces a 7-panel figure showing all processed components:

- **Dim background**: full FTAN image (including rejected regions)
- **Bright overlay**: valid FTAN region (passes far-field + SNR criteria)
- **Yellow dashed line**: far-field boundary (wavelength = distance)
- **White x**: raw picks (before QC)
- **Cyan +**: QC-filtered picks (after far-field + SNR masking)
- **Red line**: final cleaned dispersion curve (after jump removal + smoothing)

## Dependencies

- Python 3.8+
- numpy, scipy, matplotlib, pandas, h5py
- obspy (for station coordinates and geodesic distance)

## References

- Levshin, A. L., et al. (1989). Seismic Surface Waves in a Laterally Inhomogeneous Earth. Kluwer.
- Levshin, A. L., et al. (1992). Making and validating broadband surface wave dispersion measurements. Ann. Geofis., 35, 17-27.
- Nayak, A. & Thurber, C. H. (2020). Recovering multi-component Rayleigh-wave group velocities from ambient noise cross-correlations. GRL, 47.
