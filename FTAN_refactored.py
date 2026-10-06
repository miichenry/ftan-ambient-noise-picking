"""Frequency-Time ANalysis (FTAN) for surface-wave group-velocity dispersion.

This module implements the group-velocity measurement used by the mode-separation
pipeline in ``main_refactored.py``, following the classical FTAN scheme of
Levshin et al. (1989, 1992) and the multi-component Rayleigh-wave method of
Nayak & Thurber (2020).

The workhorse is :func:`calc_ftan`, a two-stage FTAN:

* **Step 1 - basic FTAN.**  A (one-sided, symmetrized) cross-correlation is passed
  through a bank of narrow Gaussian band-pass filters centred at log-spaced periods.
  For each filter the analytic-signal (Hilbert) envelope is computed and mapped from
  lag time ``t`` onto group velocity ``vg = dist / t``.  Stacking the per-period
  envelopes produces a period x group-velocity image whose bright ridge is the
  dispersion curve.

* **Step 2 - floating filter (adaptive FTAN / phase equalization).**  An *optional*
  refinement (Levshin et al. 1992): the spectrum is phase-equalized using the Step-1
  dispersion curve, then re-filtered with a narrower Gaussian plus an adaptive time
  window.  This step is opt-in via ``return_floating`` and carries an important
  caveat about the interaction between equalization and the time window - see the
  detailed note in :func:`calc_ftan`.

Conventions
-----------
* Group velocity increases toward *shorter* lag time (``vg = dist / t``).
* The returned FTAN image has its period axis in *descending* order: row ``i``
  corresponds to ``period_centers[::-1][i]``.  This lines up with the ``(T2d, vg2d)``
  meshes returned for :func:`matplotlib.pyplot.pcolormesh` and with ``T_plot`` in
  ``main_refactored.py``.
* The image is normalized by its single *global* maximum (not per period).

"""

import numpy as np
from scipy.signal import hilbert, savgol_filter
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: PHASE EQUALIZATION (Levshin et al. 1992, eq. 4)
# ══════════════════════════════════════════════════════════════════════════════

def compute_phase_equalization(tau_vec, freqs_vec):
    r"""Compute the phase-equalization function psi(omega) of Levshin et al. (1992).

    Implements equation (4) of Levshin et al. (1992):

        psi(omega) = -[ integral_0^omega tau(eta) d(eta) + c1 * omega + c2 ]

    with the integration constants set to zero (``c1 = c2 = 0``).  The equalized
    spectrum ``K'(omega) = K(omega) * exp(-i * psi(omega))`` removes the
    frequency-dependent group delay ``tau(f)``, collapsing a dispersed wave train
    back toward zero lag so it can be isolated with a narrow time window.

    Parameters
    ----------
    tau_vec : numpy.ndarray
        Group-delay curve tau(f) = dist / vg(f) in seconds, sampled on ``freqs_vec``.
    freqs_vec : numpy.ndarray
        Frequency axis in Hz (typically ``numpy.fft.rfftfreq`` output).  ``freqs_vec[0]``
        is expected to be 0 (DC).

    Returns
    -------
    numpy.ndarray
        Accumulated phase psi(omega) in radians, same shape as ``freqs_vec``.
    """
    # Guard the DC singularity: the group delay is undefined at f = 0, so borrow the
    # first finite value.  It carries zero weight in the integral (the f=0 sample is
    # the lower limit) but keeps the array well defined.
    tau_vec_safe = tau_vec.copy()
    tau_vec_safe[0] = tau_vec[1] if len(tau_vec) > 1 else tau_vec[0]

    # Cumulative trapezoidal integral of tau over frequency (Hz).  We integrate the
    # samples from index 1 onward and leave psi_integral[0] = 0 at DC.
    psi_integral = np.zeros_like(freqs_vec)
    if len(freqs_vec) > 1:
        psi_integral[1:] = cumulative_trapezoid(tau_vec_safe[1:], freqs_vec[1:], initial=0)

    # Eq. (4) with c1 = c2 = 0.  The 2*pi converts the Hz-integral to angular frequency.
    psi = -(psi_integral) * 2 * np.pi

    return psi


def validate_dispersion_curve(T_vec, vg_vec, ftan_norm_vec, threshold=0.1):
    """Filter and smooth an extracted dispersion curve.

    Keeps only the periods whose (normalized) FTAN amplitude exceeds ``threshold``,
    applies a short Savitzky-Golay smoother, and warns if the resulting curve is
    non-monotonic (a physically suspect group-velocity dispersion usually varies
    smoothly and monotonically over a limited period band).

    Parameters
    ----------
    T_vec : numpy.ndarray
        Periods (s).
    vg_vec : numpy.ndarray
        Group velocities (km/s), aligned with ``T_vec``.
    ftan_norm_vec : numpy.ndarray
        Normalized FTAN amplitude at each ``(T, vg)`` point; used only for the
        signal mask.
    threshold : float, optional
        Minimum normalized amplitude to accept a point.  Default 0.1.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Filtered/smoothed ``(T_valid, vg_valid)``.
    """
    mask_signal = ftan_norm_vec > threshold
    T_filtered = T_vec[mask_signal]
    vg_filtered = vg_vec[mask_signal]

    # Savitzky-Golay smoothing (order-2) when there are enough samples; the window
    # length must be odd and not exceed the number of points.
    if len(vg_filtered) >= 5:
        try:
            window_length = min(5, len(vg_filtered) // 2 * 2 + 1)
            vg_filtered = savgol_filter(vg_filtered, window_length, 2)
        except Exception as e:
            print(f"Warning: could not smooth dispersion curve: {e}")

    if len(vg_filtered) > 1:
        diff = np.diff(vg_filtered)
        is_monotonic = (diff > 0).all() or (diff < 0).all()
        if not is_monotonic:
            print("Warning: dispersion curve is not monotonic")

    return T_filtered, vg_filtered


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTINE: TWO-STAGE FTAN
# ══════════════════════════════════════════════════════════════════════════════

# Number of log-spaced period centres in the filter bank (matches the original code).
N_PERIOD = 120

# Amplitude threshold (fraction of the global maximum) below which a period row is
# treated as noise and excluded from picking.  Nayak & Thurber (2020) keep maxima
# above ~0.3 of the globally normalized image.  See "Normalization" note below.
SIGNAL_THRESHOLD = 0.3


def calculate_snr(signal_data, noise_data):
    """Calculate SNR in decibels.

    ``SNR_dB = 20 * log10(mean(|signal_data|) / mean(|noise_data|))``, using
    ``nanmean`` so NaNs in either window don't propagate, and a small epsilon on
    the noise mean to avoid a division by zero for a silent noise window.

    Parameters
    ----------
    signal_data, noise_data : array_like
        Sample values (e.g. envelope amplitude) from the signal and noise windows.

    Returns
    -------
    float
        SNR in dB.  ``-inf``/``nan`` are possible if a window is empty or all-NaN.
    """
    signal_mean = np.nanmean(np.abs(signal_data))
    noise_mean = np.nanmean(np.abs(noise_data)) + 1e-12  # avoid division by zero
    snr = 20 * np.log10(signal_mean / noise_mean)
    return snr


def calc_ftan(data, dist_km=153.0, alpha=5.0, alpha2=10.0, TMIN=0.9, TMAX=20.0,
              vg_min=1.0, vg_max=4.0, n_vg=500, sigma_t=5.0, dt=0.01,
              return_floating=False, equalize=False, window_mode='tau',
              verbose=False):
    r"""Two-stage Frequency-Time ANalysis of a one-sided cross-correlation.

    Parameters
    ----------
    data : numpy.ndarray
        One-sided (symmetrized) cross-correlation waveform, sampled at ``dt``.
    dist_km : float
        Inter-station distance in km.  Sets the group-velocity mapping ``vg = dist/t``,
        so it **must** match the real distance of the station pair.
    alpha : float
        Width parameter of the Step-1 Gaussian band-pass filters.  The filter is
        ``exp(-alpha * ((f - fc) / fc)^2)``; larger ``alpha`` -> narrower filter.
    alpha2 : float
        Same, but for the (narrower) Step-2 floating filter.
    TMIN, TMAX : float
        Period range (s); the filter bank is log-spaced between them.
    vg_min, vg_max : float
        Group-velocity range (km/s) of the output image axis.
    n_vg : int
        Number of samples on the group-velocity axis.
    sigma_t : float
        Base width (s) of the Step-2 adaptive time window (see ``window_mode``).
    dt : float
        Sample interval (s), normally ``st[0].stats.delta``.
    return_floating : bool, optional
        If ``False`` (default) return only the Step-1 basic FTAN image.  If ``True``,
        also compute and return the Step-2 floating-filter image.
    equalize : bool, optional
        Step-2 only.  If ``True`` apply the phase equalization of
        :func:`compute_phase_equalization` before the narrowband filter.  Default
        ``False`` - see the "Floating-filter caveat" below.
    window_mode : {'tau', 'zero', 'none'}, optional
        Step-2 only.  Where to center the adaptive Gaussian time window:
        ``'tau'`` at the expected arrival ``dist/vg`` (default), ``'zero'`` at
        ``t = 0``, or ``'none'`` for no window.
    verbose : bool, optional
        Print progress messages.  Default ``False``.

    Returns
    -------
    tuple
        If ``return_floating`` is ``False``:  ``(T2d, vg2d, ftan_image, snr_db)``.
        If ``return_floating`` is ``True``:   ``(T2d, vg2d, ftan_image, snr_db, floating_image)``.

        ``T2d``/``vg2d`` are the period/velocity edge meshes for ``pcolormesh``, and
        the image arrays are normalized to ``[0, 1]`` with the period axis descending.
        ``snr_db`` is a 1-D array (length ``N_PERIOD``, same descending period order as
        ``ftan_image``'s rows) with the per-period SNR in dB of the Step-1 narrowband
        envelope: ``20*log10(mean(|envelope| in the expected-arrival window) /
        mean(|envelope| in the trailing-coda window))`` (see :func:`calculate_snr`).
        A period is ``nan`` if either window has no samples (e.g. ``dist_km`` too
        small/large for ``[vg_min, vg_max]``, or too short a trace for a coda window).

    Notes
    -----
    **Normalization.**  The original code normalized
    *each period row* to its own maximum, which makes the plot look uniformly bright
    but destroys the amplitude information a peak-picker needs: a threshold of the
    form ``row.max() > 0`` then accepts *every* period, including pure noise.  Here
    the image is instead normalized by its single **global** maximum, so a
    fractional threshold (``SIGNAL_THRESHOLD``, ~0.3, as in Nayak & Thurber 2020) is
    meaningful and noise-only periods are rejected.  The visible trade-off is that
    genuinely weak bands look dimmer than under per-period normalization.

    **Floating-filter caveat.**  Equalization recenters the wave-train energy to ``t ~= 0`` while
    the adaptive window in the ``vg = dist/t`` readout is centered at
    ``tau = dist/vg`` (tens of seconds).  With ``equalize=True`` and
    ``window_mode='tau'`` the window therefore sits on the residual tail rather than
    the main energy (only masked by normalization).  For a self-consistent
    ``dist/t`` group-velocity readout the default is ``equalize=False`` (a narrowband
    windowed FTAN refinement).  If you want true adaptive FTAN, use
    ``equalize=True`` together with ``window_mode='zero'``.  In the original code the
    floating result was computed but never returned; here it is returned
    only when explicitly requested via ``return_floating``.
    """
    # ── Filter bank: log-spaced period centres and their frequencies ─────────────
    period_centers = np.logspace(np.log10(TMIN), np.log10(TMAX), N_PERIOD)
    freq_centers = 1.0 / period_centers

    N = len(data)
    freqs_fft = np.fft.rfftfreq(N, d=dt)
    DATA_FFT = np.fft.rfft(data, n=N)
    time_raw = np.arange(N) * dt
    t_safe = time_raw[1:]          # drop t = 0 to avoid division by zero in dist/t

    # ════════════════════════════════════════════════════════════════════════════
    # STEP 1: basic FTAN  ->  initial dispersion curve
    # ════════════════════════════════════════════════════════════════════════════
    if verbose:
        print("Step 1: basic FTAN...")

    vg_grid = np.linspace(vg_min, vg_max, n_vg)
    ftan_img = np.zeros((len(period_centers), n_vg))
    snr_db = np.full(len(period_centers), np.nan)

    # Lag time -> group velocity is the same for every period (only dist_km/t_safe),
    # so the signal/noise split by expected-arrival window is computed once.
    vg_of_t = dist_km / t_safe
    signal_mask = (vg_of_t >= vg_min) & (vg_of_t <= vg_max)   # expected arrival window
    noise_mask = vg_of_t < vg_min                             # late coda, after the
                                                                # slowest expected arrival

    for i, fc in enumerate(freq_centers):
        # Narrowband Gaussian filter centred at fc, applied in the frequency domain.
        gaussian = np.exp(-alpha * ((freqs_fft - fc) / fc) ** 2)
        filtered = np.fft.irfft(DATA_FFT * gaussian, n=N)
        # Analytic-signal envelope (instantaneous amplitude of the narrowband trace).
        envelope = np.abs(hilbert(filtered))[1:]

        # Per-period SNR (dB) of this narrowband envelope: expected-arrival window vs.
        # the trailing coda.  Computed on the same envelope used to build the FTAN
        # column, before it gets resampled onto the vg grid.
        if signal_mask.sum() >= 1 and noise_mask.sum() >= 1:
            snr_db[i] = calculate_snr(envelope[signal_mask], envelope[noise_mask])

        # Map lag time -> group velocity and resample onto the common vg grid.
        mask = signal_mask
        if mask.sum() < 2:
            continue
        # vg_of_t is monotonically decreasing in t, so reverse to make it increasing
        # for np.interp (which requires ascending x).
        vg_s = vg_of_t[mask][::-1]
        env_s = envelope[mask][::-1]
        ftan_img[i] = np.interp(vg_grid, vg_s, env_s, left=0, right=0)

    # Global normalization (see "Normalization" note in the docstring).
    ftan_norm = _normalize_global(ftan_img)

    # Dispersion curve = smoothed per-period amplitude maximum, kept only where the
    # (globally normalized) amplitude clears the signal threshold.
    ftan_smooth = uniform_filter1d(ftan_norm, size=15, axis=1)
    peak_vg = np.array([vg_grid[np.argmax(row)] for row in ftan_smooth])
    has_signal = np.array([ftan_norm[i].max() > SIGNAL_THRESHOLD
                           for i in range(len(period_centers))])

    if verbose:
        print(f"  basic FTAN done; {has_signal.sum()} periods above threshold.")

    T_plot = period_centers[::-1]
    T2d, vg2d = _build_mesh(T_plot, vg_min, vg_max, n_vg)
    snr_db_plot = snr_db[::-1]  # same descending period order as ftan_norm/T_plot

    # By default we return the (validated) Step-1 result.  Step 2 is opt-in.
    if not return_floating:
        return T2d, vg2d, ftan_norm[::-1, :], snr_db_plot

    # ════════════════════════════════════════════════════════════════════════════
    # STEP 2: floating filter (adaptive FTAN / phase equalization)
    # ════════════════════════════════════════════════════════════════════════════
    if verbose:
        print("Step 2: floating filter...")

    # Step-1 dispersion curve, ordered by increasing frequency and lightly cleaned.
    T_disp = period_centers[has_signal]
    vg_disp = peak_vg[has_signal]
    f_disp = 1.0 / T_disp
    idx = np.argsort(f_disp)
    f_disp, vg_disp, T_disp = f_disp[idx], vg_disp[idx], T_disp[idx]
    T_disp, vg_disp = validate_dispersion_curve(
        T_disp, vg_disp, peak_vg[has_signal][idx], threshold=0.05)

    if len(T_disp) < 2:
        # Not enough of a curve to build tau(f); return an empty floating image.
        if verbose:
            print("  too few periods for the floating filter; skipping.")
        return T2d, vg2d, ftan_norm[::-1, :], snr_db_plot, np.zeros_like(ftan_norm)[::-1, :]

    # tau(f) = dist / vg(f), extended over the whole FFT frequency axis by
    # constant extrapolation at the band edges.
    tau_disp = dist_km / vg_disp
    tau_interp = interp1d(1.0 / T_disp, tau_disp, bounds_error=False,
                          fill_value=(tau_disp[0], tau_disp[-1]), kind='linear')
    tau_fft = tau_interp(freqs_fft)

    # Optionally equalize the spectrum: K'(omega) = K(omega) * exp(-i * psi(omega)).
    if equalize:
        psi = compute_phase_equalization(tau_fft, freqs_fft)
        DATA_FFT_eq = DATA_FFT * np.exp(-1j * psi)
        if verbose:
            print(f"  psi range [{psi.min():.3f}, {psi.max():.3f}] rad; "
                  f"tau range [{tau_fft.min():.4f}, {tau_fft.max():.4f}] s")
    else:
        DATA_FFT_eq = DATA_FFT

    ftan_float = np.zeros((len(period_centers), n_vg))
    for i, (fc, T) in enumerate(zip(freq_centers, period_centers)):
        # Only process periods inside the measured dispersion band.
        if not (T_disp.min() <= T <= T_disp.max()):
            continue

        # Narrower Gaussian (alpha2 > alpha) for sharper period resolution.
        gaussian = np.exp(-alpha2 * ((freqs_fft - fc) / fc) ** 2)
        filtered_time = np.fft.irfft(DATA_FFT_eq * gaussian, n=N)

        # Expected group arrival time at this period.
        vg_T = float(np.interp(T, T_disp, vg_disp))
        tau_T = dist_km / vg_T

        # Adaptive Gaussian time window; the width grows with period so long-period
        # (slower, broader) arrivals are not clipped.
        sigma_t_adaptive = sigma_t * np.sqrt(T / 10.0)
        if window_mode == 'tau':
            window = np.exp(-0.5 * ((time_raw - tau_T) / sigma_t_adaptive) ** 2)
        elif window_mode == 'zero':
            window = np.exp(-0.5 * (time_raw / sigma_t_adaptive) ** 2)
        else:  # 'none'
            window = np.ones_like(time_raw)

        filtered_windowed = filtered_time * window
        envelope = np.abs(hilbert(filtered_windowed))[1:]

        vg_of_t = dist_km / t_safe
        mask = (vg_of_t >= vg_min) & (vg_of_t <= vg_max)
        if mask.sum() < 2:
            continue
        vg_s = vg_of_t[mask][::-1]
        env_s = envelope[mask][::-1]
        ftan_float[i] = np.interp(vg_grid, vg_s, env_s, left=0, right=0)

    ftan_float = _normalize_global(ftan_float)

    if verbose:
        n_float = int(np.sum([ftan_float[i].max() > SIGNAL_THRESHOLD
                              for i in range(len(period_centers))]))
        print(f"  floating filter done; {n_float} periods above threshold.")

    return T2d, vg2d, ftan_norm[::-1, :], snr_db_plot, ftan_float[::-1, :]


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_global(img):
    """Scale an FTAN image by its single global maximum (no-op if all zeros)."""
    mx = img.max()
    if mx > 0:
        return img / mx
    return img


def _build_mesh(T_arr, vg_min, vg_max, n_vg):
    """Build ``pcolormesh`` edge meshes for a (descending) period axis.

    ``pcolormesh(shading='flat')`` needs cell *edges*, one more than the number of
    cell centres along each axis.  Period edges are placed at geometric midpoints
    between neighbouring centres (appropriate for a log-spaced axis), with the two
    outer edges extrapolated geometrically.  Velocity edges are linearly spaced.
    """
    T_edges = np.concatenate([
        [T_arr[0] * (T_arr[0] / T_arr[1])],
        np.sqrt(T_arr[:-1] * T_arr[1:]),
        [T_arr[-1] * (T_arr[-1] / T_arr[-2])]
    ])
    vg_edges = np.linspace(vg_min, vg_max, n_vg + 1)
    return np.meshgrid(T_edges, vg_edges)