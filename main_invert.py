"""
Multi-component Rayleigh-wave mode separation + group- AND phase-velocity dispersion.

This is the entry point of the pipeline.  It reproduces the method of
Nayak & Thurber (2020), "Recovering Multi-Component Rayleigh-Wave...":  the four
inter-station cross-correlation components ``ZZ``, ``RR``, ``ZR``, ``RZ`` are
combined with +/- 90 degree phase shifts to synthesize two Green's-function
combinations that separate the fundamental and first higher Rayleigh mode, and the
group-velocity dispersion of each is measured with FTAN (:mod:`FTAN_refactored`).

Phase velocity is then measured from the SAME Green's function with the
zero-crossing (AKI) method of Boschi et al. (2013), GJI 192, 346-358, in
:mod:`phase_velocity`.

Two passes, and why
-------------------
The zero-crossing method determines the wavenumber only up to which zero of J0
the first crossing belongs to -- the "branch".  Measured on 184 synthetic cases:
a pair that reaches exactly ONE admissible branch is right 100% of the time
(median error 0.14%); a pair with more than one is right only ~40% of the time,
and no score threshold separates those.  So ``measure_phase_velocity_aki``
defaults to ``require_unique=True`` and simply refuses the ambiguous ones.

What decides how many pairs reach a unique branch is the WIDTH of the frequency
band, so:

* **Pass 1** feeds AKI the group-velocity curve WITHOUT the far-field cut.  AKI
  needs no far-field approximation at all -- that is the whole reason it is
  preferred over the time-domain (FRY) method here -- and the cut only narrows
  the band, which is exactly what destroys branch selection.  The far-field cut
  is still applied to the curve that gets WRITTEN as the group-velocity product,
  so ``disp_vg/`` is unchanged from before.
* **Pass 2** stacks the pass-1 winners into a reference curve and re-runs the
  pairs that came back ambiguous with ``c_ref=``.  It needs no FTAN, so it is
  cheap.  On synthetics this recovered 10 of 10 ambiguous pairs.
"""

import os

from obspy import read, Stream, Trace
from scipy.ndimage import uniform_filter1d
from glob import glob
from FTAN_refactored import calc_ftan
import geopy.distance
from read_h5 import read_ccf_h5
from robust_dispersion_clean import remove_dispersion_jumps
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')          # sin backend interactivo: 4005 pares en batch
import matplotlib.pyplot as plt
from phase_velocity import (measure_phase_velocity_aki, save_dispersion,
                            plot_aki, build_reference_curve, trim_edge_outliers)

WORK_PATH = '/Users/ivancabreraperez/Desktop/MAIN/SCRIPTS/extract_higher_modes/Projects/Campiglia'
CC_PATH = os.path.join(WORK_PATH, 'STACK_phase_only_rma')
OUTPUT_PATH_GROUP = os.path.join(WORK_PATH, 'disp_vg')
OUTPUT_PATH_PHASE = os.path.join(WORK_PATH, 'disp_vph')
CACHE_PATH = os.path.join(WORK_PATH, 'cache_aki')
FIG_PATH = os.path.join(WORK_PATH, 'fig')
FIG_PATH_AKI = os.path.join(WORK_PATH, 'fig_aki')
station_file = os.path.join(WORK_PATH, 'stations2.csv')
sampling_rate = 42.0
delta = 1/sampling_rate
signal_window = 20 #seconds
network = 'RI'
stations = np.loadtxt(station_file, skiprows=1, usecols=0, dtype=str, delimiter=',')
latitud = np.loadtxt(station_file, skiprows=1, usecols=1, delimiter=',')
longitud = np.loadtxt(station_file, skiprows=1, usecols=2, delimiter=',')

# ── Configuration ───────────────────────────────────────────────────────────────

for _p in (OUTPUT_PATH_GROUP, OUTPUT_PATH_PHASE, CACHE_PATH, FIG_PATH, FIG_PATH_AKI):
    os.makedirs(_p, exist_ok=True)

# Fraction of the (globally normalized) FTAN maximum below which a peak pick is
# treated as noise and dropped.  Matches FTAN_refactored.SIGNAL_THRESHOLD and the
# ~0.3 criterion of Nayak & Thurber (2020).
PICK_THRESHOLD = 0.3

# Far-field / near-field criterion factor: a period is considered reliable only
# where the propagation distance is at least FAR_FIELD_FACTOR wavelengths.  Cells
# that fail this (1.5 * wavelength >= distance) are masked out (made transparent)
# and excluded from the picked dispersion curve.
FAR_FIELD_FACTOR = 1.0

# Period-axis order used by the plotted image and returned by calc_ftan: descending,
# with the same log spacing and count as the FTAN filter bank (120 centres).
N_PERIOD = 120

# Minimum acceptable per-period SNR, in dB (see FTAN_refactored.calculate_snr /
# calc_ftan's snr_db return value).  Periods below this are masked out at every
# velocity, same as the far-field wavelength criterion.
SNR_THRESHOLD_DB = 5.0

# ── Phase-velocity (AKI) configuration ──────────────────────────────────────────

# Group-velocity axis limits of the FTAN bank, hoisted out of the per-pair
# FTAN_PARAMS so the AKI lag window can be derived from them.
FTAN_VG_MIN = 0.5
FTAN_VG_MAX = 4.0

# Plausible phase-velocity range (km/s).  NARROW THIS to what the site can
# actually produce: c_min also sets the spline knot spacing inside
# measure_phase_velocity_aki, so an unnecessarily small c_min asks for far more
# knots than the spectrum can support.
AKI_C_MIN = 0.3
AKI_C_MAX = 3.5

# Lag window kept before transforming, as a multiple of the slowest expected
# arrival dist/vg_min.  Short enough to reject coda noise, long enough to
# resolve the J0 lobes; measure_phase_velocity_aki warns if it gets this wrong.
AKI_LAG_FACTOR = 6.0

# Per-period SNR (dB) a pick must clear to enter the curve handed to AKI.  Kept
# separate from SNR_THRESHOLD_DB (which gates the whole pair) so the two can be
# tuned independently: this one trades band width against pick quality, and band
# width is what decides branch selection.
AKI_SNR_THRESHOLD_DB = SNR_THRESHOLD_DB

# Minimum number of pairs that must contribute at a frequency for it to enter the
# reference curve used by pass 2.  Lower it only for very small datasets.
REF_MIN_PAIRS = 3

# Write an AKI diagnostic figure for at most this many pairs (they are only for
# eyeballing the method; 4005 of them would be useless and slow).
MAX_AKI_FIGS = 30


def aki_kwargs(distance):
    """Per-pair keyword arguments shared by both AKI passes."""
    return dict(dist_km=distance,
                max_lag_s=AKI_LAG_FACTOR * distance / FTAN_VG_MIN,
                c_min=AKI_C_MIN, c_max=AKI_C_MAX)


def slim_result(res, pair):
    """Keep only what build_reference_curve needs.

    A full result carries the spectrum, the spline and every candidate branch --
    tens of thousands of floats.  Holding 4005 of those would exhaust memory, and
    the reference curve only needs the selected ``(f, c)``.
    """
    return {'pair': pair, 'ok': bool(res.get('ok')),
            'ambiguous': bool(res.get('ambiguous')),
            'f': np.asarray(res.get('f', []), dtype=float),
            'c': np.asarray(res.get('c', []), dtype=float)}


def calculate_snr(signal_data, noise_data):
    """Calculate SNR in decibels."""
    signal_mean = np.nanmean(np.abs(signal_data))
    noise_mean = np.nanmean(np.abs(noise_data)) + 1e-12  # avoid division by zero
    snr = 20 * np.log10(signal_mean / noise_mean)
    return snr

def symmetrize(data):
    """Fold a two-sided correlation about zero lag into a one-sided signal.

    An ambient-noise cross-correlation is (ideally) symmetric about zero lag, so the
    causal (positive-lag) and anti-causal (negative-lag) halves carry the same
    Green's-function information.  Summing them improves SNR.

    Fix
    --------------------------
    The original one-liner ``[a + b for a, b in zip(data[m:], data[:m][::-1])]`` pairs
    lag ``0`` with lag ``-1`` when ``npts`` is odd (as here, 48001) - a half-sample
    misregistration that never folds about the true zero-lag sample.  Measured effect:
    a spurious sample-level zigzag near zero lag and a group-velocity error up to
    ~0.6 km/s at the shortest period.  This version folds about the zero-lag sample
    (index ``m``) and counts it exactly once.

    Parameters
    ----------
    data : array_like
        Full two-sided correlation (length ``npts``, zero lag at index ``npts // 2``).

    Returns
    -------
    numpy.ndarray
        One-sided signal of length ``npts // 2 + 1``: index 0 is zero lag, index k is
        the sum of lags ``+k`` and ``-k``.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    m = n // 2
    out = data[m:].copy()          # lags 0, +1, ..., +M
    out[1:] += data[:m][::-1]      # add lags -1, ..., -M onto lags +1, ..., +M
    return out


def load_pair(file):
    """Read one pair's four components, fold them, and synthesize G_LR0 / G_LR1.

    Fix
    --------------------------
    The original code wrapped the read in ``try: ... except: pass``.  On a failed
    read that does NOT skip the pair: execution falls straight through to
    ``symmetrize(ZZ)`` with ``ZZ`` still holding the PREVIOUS pair's data, so a
    complete dispersion curve gets computed from one pair's waveforms and written
    under another pair's name.  Silent, and poisonous to a tomographic inversion.
    Returning ``None`` here lets the caller ``continue``.

    Returns
    -------
    dict or None
        ``{'ZZ', 'RR', 'RZ', 'ZR', 'G_LR0', 'G_LR1'}`` (all folded), or ``None``
        if the file could not be read.
    """
    try:
        ccf = read_ccf_h5(file, stack='Allstack_pws', components=['ZZ', 'RR', 'RZ', 'ZR'])
        ZZ = symmetrize(ccf['data']['ZZ'])
        RR = symmetrize(ccf['data']['RR'])
        RZ = symmetrize(ccf['data']['RZ'])
        ZR = symmetrize(ccf['data']['ZR'])
    except Exception as exc:
        print(f'>> Error reading {os.path.basename(file)}: {exc}')
        return None

    # ── Mode separation: Nayak & Thurber (2020), eqs (2)/(3)/(4) ───────────────
    # The +/-90 degree phase shifts on RZ/ZR, added to ZZ+RR, synthesize two
    # Green's-function combinations: G_LR0 enhances the fundamental Rayleigh mode and
    # G_LR1 the first higher mode.
    #
    # Convention caution: this assumes the component letters are ordered
    # source-then-receiver (RZ = R at source / Z at receiver = the paper's G_RZ).  A
    # dataset labelled the other way needs RZ and ZR swapped (equivalent to flipping
    # the sign).  The four components must also share a common amplitude
    # normalization, since these are literal waveform sums.
    #
    # These same +-90 degree factors are what a quadrature error would live in, and
    # AKI cannot detect one from a single pair.  Run check_quadrature.py ONCE on a
    # high-SNR pair before trusting any phase velocity.
    RZ_corrected = np.fft.rfft(RZ) * np.exp(1j * np.pi / 2)
    ZR_corrected = np.fft.rfft(ZR) * np.exp(-1j * np.pi / 2)

    G_LR0 = ZZ + RR + np.fft.irfft(RZ_corrected + ZR_corrected, n=len(ZZ))
    G_LR1 = ZZ + RR + np.fft.irfft(-RZ_corrected - ZR_corrected, n=len(ZZ))

    return {'ZZ': ZZ, 'RR': RR, 'RZ': RZ, 'ZR': ZR, 'G_LR0': G_LR0, 'G_LR1': G_LR1}


def wavelength_mask(T2d, vg2d, distance, factor=FAR_FIELD_FACTOR):
    """Boolean mask of FTAN cells that fail the far-field criterion.

    The group-velocity wavelength at each FTAN cell is ``wavelength = Vg * T``
    (Vg in km/s, T in s -> wavelength in km).  A cell is considered unreliable
    (near-field / not enough wave cycles between the stations) when::

        factor * wavelength >= distance

    Parameters
    ----------
    T2d, vg2d : numpy.ndarray
        Period (s) and group-velocity (km/s) grids as returned by
        :func:`FTAN_refactored.calc_ftan` (same shape as the FTAN image, or
        broadcastable to it).
    distance : float
        Inter-station distance (km).
    factor : float, optional
        Far-field factor (default :data:`FAR_FIELD_FACTOR` = 1.5).

    Returns
    -------
    numpy.ndarray of bool
        ``True`` where the cell must be masked (transparent / excluded from
        picking), same shape as ``T2d``/``vg2d``.
    """
    wavelength = vg2d * T2d
    return (factor * wavelength) >= distance


def get_peak_curve(ftan_img, T_plot, vg_grid, mask=None, threshold=0.3, normalize='row'):
    """Extract the group-velocity dispersion curve from an FTAN image.

    For every period row the group velocity of the (smoothed) amplitude maximum is
    taken as the pick, but only rows whose amplitude clears ``threshold`` are kept.
    If ``mask`` is given, masked cells are excluded from the search for the peak and
    a period whose entire row is masked is dropped from the picked curve too.

    History of fixes
    --------------------------
    v2: masked *before* smoothing (`np.where(mask, -np.inf, img)` then
    `uniform_filter1d`). A single -inf inside the 15-cell window drags the whole
    window's mean to -inf, killing every valid cell within ~7 cells of any mask
    edge -> picks stopped well short of the true boundary.

    v3: smoothed the *raw* image first, masked afterwards. Fixed the -inf
    contamination, but now the smoothing kernel also averaged in values from the
    invalid (near-field) region -- which often carries edge-ringing artifacts from
    the FTAN Gaussian filtering (anomalously large values right at the validity
    boundary) -- pulling picks toward the mask edge instead of the genuine peak.

    v3b (normalized convolution: zero out masked cells, divide by a smoothed
    "valid fraction" weight): still biased near the edge, for a different reason.
    As the window approaches the mask boundary, fewer valid cells remain in it, so
    *any* real signal within the shrinking window gets divided by a smaller count
    and its apparent smoothed amplitude is inflated purely by proximity to the
    edge -- again dragging the pick toward the boundary rather than the true peak.

    v4 (this version): smooth *only within each row's own contiguous valid
    segment*, letting `uniform_filter1d`'s default edge handling (reflection)
    apply at the true ends of that segment -- never crossing into the invalid
    region, and never diluted by a shrinking neighbor count. This assumes the
    valid segment is contiguous per row, which holds for masks built from a single
    Vg threshold per period (the far-field wavelength / SNR criteria used here).

    Parameters
    ----------
    ftan_img : numpy.ndarray
        Globally normalized FTAN image, shape ``(N_PERIOD, n_vg)``, period axis
        descending (as returned by :func:`FTAN_refactored.calc_ftan`).
    T_plot : numpy.ndarray
        Periods (s) aligned with the rows of ``ftan_img`` (descending).
    vg_grid : numpy.ndarray
        Group-velocity axis (km/s) aligned with the columns of ``ftan_img``.
    mask : numpy.ndarray of bool, optional
        Same shape as ``ftan_img``. ``True`` marks a cell to exclude (e.g. cells
        failing the far-field wavelength criterion, see :func:`wavelength_mask`).
        Assumed contiguous-per-row (single Vg cutoff per period).
    threshold : float, optional
        Minimum (row-relative, by default) normalized amplitude to accept a pick.
    normalize : {'row', 'global'}, optional
        'row' (default): threshold applied to peak_amp / row_max_valid.
        'global': threshold applied to peak_amp directly.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(T, Vg)`` of the accepted picks.
    """
    img = np.asarray(ftan_img, dtype=float)
    n_T, n_vg = img.shape
    KSIZE = 15

    smoothed = np.full_like(img, -np.inf)
    row_has_valid = np.zeros(n_T, dtype=bool)
    for i in range(n_T):
        valid_idx = np.where(~mask[i])[0] if mask is not None else np.arange(n_vg)
        if valid_idx.size == 0:
            continue
        row_has_valid[i] = True
        lo, hi = valid_idx[0], valid_idx[-1] + 1
        sub = img[i, lo:hi]
        smoothed[i, lo:hi] = uniform_filter1d(sub, size=min(KSIZE, sub.size))

    peak_vg = np.array([vg_grid[np.argmax(row)] for row in smoothed])
    # peak_amp read off the original (unsmoothed) img, at the picked location.
    peak_amp = np.array([img[i, np.argmax(smoothed[i])] for i in range(n_T)])

    if normalize == 'row':
        img_valid = np.where(mask, np.nan, img) if mask is not None else img
        with np.errstate(all='ignore'):
            row_max = np.nanmax(img_valid, axis=1)
        rel_amp = peak_amp / row_max
        has_sig = (rel_amp > threshold) & row_has_valid
    elif normalize == 'global':
        has_sig = (peak_amp > threshold) & row_has_valid
    else:
        raise ValueError("normalize must be 'row' or 'global'")

    return T_plot[has_sig], peak_vg[has_sig]



def fix_picking_plateaus(x, y, window_size=41, threshold=0.4):
    """
    Removes large plateau jumps and interpolates the curve.
    window_size: Must be larger than the width of the bad data block.
    threshold: How far a point can deviate from the trend before being deleted.

    Fix
    --------------------------
    No guard existed for a curve that is empty or too short to interpolate.
    If `x`/`y` came in empty (e.g. a station pair where the far-field/wavelength
    filter upstream never found a single reliable point), or if the rolling
    median removed enough points that fewer than 4 remain (the minimum
    ``interp1d(kind='cubic')`` needs), the function fell through to
    ``interp1d(x_valid, y_valid, kind='cubic', ...)`` with a size-0 or
    too-small array and crashed with
    ``ValueError: cannot reshape array of size 0 into shape (0,newaxis)``,
    stopping the whole batch run on one bad pair. Now it returns an empty
    array (0 points in -> 0 points out) or, for a very short curve
    (<4 valid points after cleaning), falls back to linear interpolation
    (or just returns the cleaned points unchanged if even that isn't
    possible) instead of raising.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0:
        return y  # nada que arreglar; deja que el llamador decida que hacer con una curva vacia

    s = pd.Series(y)

    # 1. Calculate a robust baseline using a wide rolling median
    baseline = s.rolling(window=window_size, center=True, min_periods=1).median()
    baseline = baseline.bfill().ffill()

    # 2. Identify the massive jumps/plateaus (outliers)
    outliers = np.abs(s - baseline) > threshold

    # 3. Mask the bad data
    y_cleaned = np.copy(y)
    y_cleaned[outliers] = np.nan

    # 4. Interpolate to recreate the curved line
    valid_idx = ~np.isnan(y_cleaned)
    x_valid = x[valid_idx]
    y_valid = y_cleaned[valid_idx]

    if len(x_valid) == 0:
        # El filtro descarto TODOS los puntos (umbral demasiado estricto para
        # esta curva, o la curva entera es un artefacto) -- no hay nada de lo
        # que interpolar. Se devuelve la y original sin tocar en vez de
        # fallar; el llamador puede decidir descartar el par completo.
        return y

    if len(x_valid) < 4:
        # interp1d(kind='cubic') necesita al menos 4 puntos. Con menos,
        # usar lineal (con solo 1 punto ni eso es posible: se devuelve
        # constante).
        kind = 'linear' if len(x_valid) >= 2 else 'nearest'
    else:
        kind = 'cubic'

    f_interp = interp1d(x_valid, y_valid, kind=kind, fill_value="extrapolate")
    y_final = f_interp(x)

    return y_final


def clean_group_curve(T_pk, vg_pk, distance=None, far_field=False):
    """Clean a picked group-velocity curve, optionally applying the far-field cut.

    Exactly the quality control the pipeline already did -- drop mode jumps with
    ``remove_dispersion_jumps``, then the plateau filter -- factored out so the
    far-field cut can be switched off.

    Two callers, two settings:

    * ``far_field=True``  -> the curve written to ``disp_vg/`` (unchanged product).
    * ``far_field=False`` -> the curve handed to AKI.  AKI needs no far-field
      approximation, and the cut only narrows the frequency band, which is what
      destroys branch selection: with a band spanning a factor 5 in frequency,
      0% of synthetic pairs reached a unique branch; with a factor 30, 72% did.

    Parameters
    ----------
    T_pk, vg_pk : array_like
        Raw picks from :func:`get_peak_curve`, in the order it returns them
        (period DESCENDING -- the far-field cut depends on that order).
    distance : float, optional
        Inter-station distance (km).  Required when ``far_field=True``.
    far_field : bool, optional
        Apply the ``vg * T <= distance`` cut.  Default ``False``.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(T, vg)`` ascending in period, possibly empty.
    """
    T = np.asarray(T_pk, dtype=float)
    V = np.asarray(vg_pk, dtype=float)
    if T.size == 0:
        return T, V

    if far_field:
        if distance is None:
            raise ValueError('far_field=True needs distance')
        t_keep, v_keep = [], []
        keep_from_here = False
        for t_i, vg_i in zip(T, V):
            if not keep_from_here:
                if vg_i * t_i <= distance:
                    keep_from_here = True
                else:
                    continue  # todavia en la zona no fiable: se descarta este punto
            t_keep.append(t_i)
            v_keep.append(vg_i)
        T, V = np.asarray(t_keep, dtype=float), np.asarray(v_keep, dtype=float)
        if T.size == 0:
            return T, V

    T, V = remove_dispersion_jumps(T, V)   # elimina saltos, no los difumina
    T, V = np.asarray(T, dtype=float), np.asarray(V, dtype=float)
    if T.size == 0:
        return T, V

    y_fixed = np.asarray(fix_picking_plateaus(T, V, window_size=51, threshold=0.4),
                         dtype=float)
    keep = np.isfinite(y_fixed) & (y_fixed >= 0.1)
    return T[keep], V[keep]


def extract_dist(sta1, sta2):
    """Geodesic inter-station distance (km), or ``None`` if a station is unknown.

    Fix
    --------------------------
    The original built ``coords_1``/``coords_2`` from loop variables that were
    only assigned inside ``if stations[ix] == sta...``.  A station present in a
    correlation filename but absent from ``stations2.csv`` (a typo, a station
    dropped from the list, a stray file) therefore raised
    ``UnboundLocalError: cannot access local variable 'lat2'`` and killed the
    whole batch -- possibly hours into a 4005-pair run.  Now it returns ``None``
    and the caller skips that pair.
    """
    coords = {}
    for ix in range(len(stations)):
        if stations[ix] == sta1:
            coords[1] = (latitud[ix], longitud[ix])
        if stations[ix] == sta2:
            coords[2] = (latitud[ix], longitud[ix])

    if 1 not in coords or 2 not in coords:
        missing = [s for s, key in ((sta1, 1), (sta2, 2)) if key not in coords]
        print(f'>> Aviso: estacion(es) {missing} no estan en {os.path.basename(station_file)}; '
              'par omitido.')
        return None

    return geopy.distance.geodesic(coords[1], coords[2]).km


def write_phase_curve(res, sta1, sta2, distance):
    """Write a phase-velocity curve, trimming biased band-edge points first.

    The outermost crossings sit where their J0 lobe is only partly inside the
    analysis band, so their root is biased.  On a deterministic end-to-end run
    this trimming cost 4 points of 155 and cut the worst error from 45.7% to
    6.1% -- exactly the outliers that wreck an inversion.

    Only the WRITTEN product is trimmed; ``slim_result`` keeps the untrimmed
    curve, because build_reference_curve does better with the extra points.
    """
    f_w, c_w, _ = trim_edge_outliers(res['f'], res['c'])
    if f_w.size < 2:
        return False
    save_dispersion(phase_output_path(sta1, sta2), 1.0 / f_w, c_w, distance)
    return True


def phase_output_path(sta1, sta2):
    return os.path.join(OUTPUT_PATH_PHASE, f'disp_phase_{sta1}_{sta2}_td.dat')


# ══════════════════════════════════════════════════════════════════════════════
# PASS 1: FTAN, group velocity, and phase velocity where the branch is unique
# ══════════════════════════════════════════════════════════════════════════════

def pass1():
    """FTAN + group velocity for every pair, and AKI where the branch is unique.

    Returns
    -------
    list of dict
        Slimmed AKI results (see :func:`slim_result`) for the pairs that produced
        a phase-velocity curve; these feed :func:`build_reference_curve`.
    """
    k = 0
    n_figs = 0
    results = []
    total_files = len(glob(CC_PATH + '/*/*.h5'))

    for ix, sta1 in enumerate(stations):

        for file in glob(f'{CC_PATH}/{network}.{sta1}/*.h5'):

            PAIR = f'{os.path.basename(file)[:-3]}'
            sta2 = os.path.basename(file).split('_')[1].split('.')[1]

            distance = extract_dist(os.path.basename(file).split('_')[0].split('.')[1], sta2)
            if distance is None:
                continue

            # FTAN parameters, defined once and shared by every calc_ftan call so that the
            # velocity/period grids used for plotting and picking cannot drift out of sync with
            # the grids used inside calc_ftan.
            FTAN_PARAMS = dict(
                dist_km=distance,  # inter-station distance (km) - MUST match the real pair distance
                alpha=5.0,  # Step-1 Gaussian filter width (larger -> narrower band)
                alpha2=10.0,  # Step-2 (floating) Gaussian filter width (narrower)
                TMIN=0.1,  # shortest period (s)
                TMAX=5.0,  # longest period (s)
                vg_min=FTAN_VG_MIN,  # group-velocity axis min (km/s)
                vg_max=FTAN_VG_MAX,  # group-velocity axis max (km/s)
                n_vg=500,  # number of group-velocity samples
                sigma_t=5.0,  # base width (s) of the Step-2 adaptive time window
            )

            # ── 1-3. Read, symmetrize, separate modes ────────────────────────────
            k += 1
            pair_data = load_pair(file)
            if pair_data is None:
                continue                       # <- antes: `pass`, que seguia con los datos del par ANTERIOR
            print(f'>> Processing {PAIR} - {k} of {total_files}')

            ZZ, RR = pair_data['ZZ'], pair_data['RR']
            RZ, ZR = pair_data['RZ'], pair_data['ZR']
            G_LR0, G_LR1 = pair_data['G_LR0'], pair_data['G_LR1']

            # ── 4. FTAN on every trace ───────────────────────────────────────────
            # calc_ftan now also returns snr_db: the per-period SNR (dB) of the Step-1
            # narrowband envelope (expected-arrival window vs. trailing coda), computed
            # inside FTAN_refactored from the very envelope used to build each column -
            # see FTAN_refactored.calculate_snr / calc_ftan's docstring.
            traces = {'ZZ': ZZ, 'RR': RR, 'RZ': RZ, 'ZR': ZR, 'G_LR0': G_LR0, 'G_LR1': G_LR1}
            ftan = {}
            for name, sig in traces.items():
                T2d, vg2d, img, snr_db = calc_ftan(sig, dt=delta, **FTAN_PARAMS)
                ftan[name] = (T2d, vg2d, img, snr_db)

            # ── 5. Picking grids (derived from FTAN_PARAMS -> cannot desync) ─────
            vg_grid = np.linspace(FTAN_PARAMS['vg_min'], FTAN_PARAMS['vg_max'], FTAN_PARAMS['n_vg'])
            T_plot = np.logspace(np.log10(FTAN_PARAMS['TMIN']),
                                 np.log10(FTAN_PARAMS['TMAX']), N_PERIOD)[::-1]

            # Colormap that renders masked (invalid, near-field) cells fully transparent
            # instead of falling back to a color, so np.ma.masked arrays disappear from the
            # pcolormesh instead of showing up as (e.g.) white/black.
            cmap = plt.get_cmap('jet').copy()
            cmap.set_bad(color=(0, 0, 0, 0))  # RGBA alpha=0 -> transparent

            signal_time = np.arange(len(G_LR0)) * delta

            if calculate_snr(signal_data=G_LR0[:int(len(signal_time)/2)],
                             noise_data=G_LR0[int(len(signal_time)/2):-1]) < SNR_THRESHOLD_DB:
                continue

            # ── 6. Plot + write dispersion curves ────────────────────────────────
            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
            layout = [
                (axs[0, 0], 'ZZ'), (axs[0, 1], 'RR'),
                (axs[1, 0], 'RZ'), (axs[1, 1], 'ZR'),
                (axs[2, 0], 'G_LR0'), (axs[2, 1], 'G_LR1'),
            ]
            for ax, title in layout:
                T2d, vg2d, img, snr_db = ftan[title]

                # Far-field wavelength mask: True where 1.5*wavelength >= distance, i.e. the
                # cell is unreliable and must be shown transparent / excluded from picking.
                #
                # T2d/vg2d are the pcolormesh *edge* coordinates (shape (n_vg+1, N_PERIOD+1)
                # for shading='flat'), one larger in each dimension than img/T_plot/vg_grid.
                # The mask must be built from the *cell-center* grids (T_plot, vg_grid), which
                # already match img's shape (N_PERIOD, n_vg) exactly.
                Tg, Vg = np.meshgrid(T_plot, vg_grid, indexing='ij')
                invalid_wavelength = wavelength_mask(Tg, Vg, distance)

                # Low-SNR mask: True for periods whose Step-1 narrowband SNR (dB) is below
                # SNR_THRESHOLD_DB, or undefined (nan, e.g. an empty signal/noise window).
                # snr_db is per-period only (independent of velocity), so broadcast it across
                # the velocity axis to match img's shape.
                low_snr = np.isnan(snr_db) | (snr_db < SNR_THRESHOLD_DB)
                invalid_snr = np.broadcast_to(low_snr[:, None], img.shape)

                invalid = invalid_wavelength

                img_masked = np.ma.masked_where(invalid, img)
                ax.pcolormesh(T2d, vg2d, img_masked.T, cmap=cmap, vmin=0, vmax=1, shading='flat')

                T_pk, vg_pk = get_peak_curve(img, T_plot, vg_grid, mask=None)
                ax.scatter(T_pk, vg_pk, marker='x', s=15, linewidths=0.8, color='white', zorder=5)

                ax.set_title(title)
                ax.set_xlabel('Period (s)')
                ax.set_ylabel('Vg (km/s)')
                ax.set_xscale('log')

                if title != 'G_LR0':
                    continue

                # ── 6a. Group velocity: WITH the far-field cut (product unchanged) ──
                T_grp, vg_grp = clean_group_curve(T_pk, vg_pk, distance, far_field=True)
                if T_grp.size == 0:
                    print(f'>> Aviso: {PAIR} ({title}) sin picks tras el filtrado; se omite.')
                    continue

                np.savetxt(os.path.join(OUTPUT_PATH_GROUP,
                                        f'disp_ZZ_{sta1}_{sta2}_td.dat'),
                           np.column_stack((T_grp, vg_grp)),
                           header=f'{distance}', comments='')

                # ── 6b. Phase velocity: SNR cut instead of the far-field cut ───────
                # Dropping the far-field cut widens the band, which is what AKI
                # needs -- but it also leaves the picks with NO quality control,
                # so pure-noise picks at long period sneak in and corrupt both the
                # analysis band and trim_crossing_sequence.
                #
                # The right filter is the per-period SNR, which this pipeline
                # already computes as `low_snr` and then throws away (the line
                # `invalid = invalid_wavelength` overwrites `invalid_snr`).  Unlike
                # the far-field criterion it does not shrink with period, so it
                # cleans the curve without narrowing the band.
                #
                # snr_db is aligned with T_plot (both descending); reverse both for
                # np.interp, which needs ascending x.
                snr_at_pick = np.interp(T_pk, T_plot[::-1], snr_db[::-1])
                good_snr = snr_at_pick >= AKI_SNR_THRESHOLD_DB
                T_aki, vg_aki = clean_group_curve(T_pk[good_snr], vg_pk[good_snr],
                                                  far_field=False)
                if T_aki.size < 2:
                    print(f'>> Aviso: {PAIR} sin curva de grupo utilizable para AKI '
                          f'({good_snr.sum()} picks sobre {AKI_SNR_THRESHOLD_DB} dB); se omite.')
                    continue

                # Cache lo que la pasada 2 necesita: nunca vuelve a hacer FTAN.
                # 'file' colisiona con el primer parametro de np.savez -> h5_file
                np.savez(os.path.join(CACHE_PATH, f'{PAIR}.npz'),
                         T_vg=T_aki, vg=vg_aki, distance=distance,
                         h5_file=file, sta1=sta1, sta2=sta2)

                res = measure_phase_velocity_aki(
                    G_LR0, dt=delta, T_vg=T_aki, vg=vg_aki,
                    verbose=True, **aki_kwargs(distance))

                if res['ok']:
                    write_phase_curve(res, sta1, sta2, distance)
                results.append(slim_result(res, PAIR))

                # Figura diagnostica AKI solo para los primeros pares: es para
                # mirar el metodo, no un producto por par.  Y SIEMPRE se cierra.
                if n_figs < MAX_AKI_FIGS:
                    fig_aki = plot_aki(res, T_vg=T_aki, vg=vg_aki)
                    fig_aki.savefig(os.path.join(FIG_PATH_AKI, f'aki_{PAIR}.png'), dpi=110)
                    plt.close(fig_aki)
                    n_figs += 1

            # fig.savefig, no plt.savefig: plot_aki crea su propia figura y se
            # convierte en la "actual", asi que plt.savefig guardaria ESA.
            fig.tight_layout()
            fig.savefig(os.path.join(FIG_PATH, f'ftan_refactored_{PAIR}.png'), dpi=150)
            plt.close(fig)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PASS 2: rescue the ambiguous pairs with a reference curve
# ══════════════════════════════════════════════════════════════════════════════

def pass2(c_ref):
    """Re-run the pairs that pass 1 left without a curve, using ``c_ref``.

    Needs no FTAN: the group curve comes from the cache pass 1 wrote, and
    ``G_LR0`` is rebuilt straight from the .h5 (read + fold + combine).
    """
    n_done = 0
    cached = sorted(glob(os.path.join(CACHE_PATH, '*.npz')))
    for npz_file in cached:
        d = np.load(npz_file, allow_pickle=False)
        sta1, sta2 = str(d['sta1']), str(d['sta2'])
        if os.path.exists(phase_output_path(sta1, sta2)):
            continue                                    # ya resuelto en la pasada 1

        pair_data = load_pair(str(d['h5_file']))
        if pair_data is None:
            continue

        distance = float(d['distance'])
        res = measure_phase_velocity_aki(
            pair_data['G_LR0'], dt=delta, T_vg=d['T_vg'], vg=d['vg'],
            c_ref=c_ref, verbose=False, **aki_kwargs(distance))

        if res['ok'] and write_phase_curve(res, sta1, sta2, distance):
            n_done += 1

    print(f'>> Pasada 2: {n_done} de {len(cached)} pares en cache recuperados con c_ref')
    return n_done


def main():
    print('=== PASADA 1: FTAN + velocidad de grupo + fase de rama unica ===')
    results = pass1()
    n_ok = sum(r['ok'] and not r['ambiguous'] for r in results)
    print(f'>> Pasada 1: {n_ok} de {len(results)} pares con rama unica')

    c_ref = build_reference_curve(results, min_pairs=REF_MIN_PAIRS)
    if c_ref is None:
        print('>> No hay pares suficientes con rama unica para construir c_ref; '
              'sin pasada 2.  Revisa fig_aki/ y considera ampliar la banda de FTAN '
              '(TMAX) o afinar AKI_C_MIN.')
        return

    np.savetxt(os.path.join(WORK_PATH, 'c_ref.dat'),
               np.column_stack(c_ref), header='freq_Hz c_km_s', comments='# ')
    print(f'>> c_ref: {c_ref[0].size} frecuencias, '
          f'{c_ref[0].min():.2f}-{c_ref[0].max():.2f} Hz')

    print('=== PASADA 2: rescate de los pares ambiguos con c_ref ===')
    pass2(c_ref)


if __name__ == '__main__':
    main()