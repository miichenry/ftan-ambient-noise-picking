"""
Velocidad de FASE de ondas de Rayleigh a partir de la MISMA funcion de Green
que ya se usa para la velocidad de grupo (ZZ, RR, G_LR0, G_LR1 de
``main_invert.py``).

Implementa los dos algoritmos de

    Boschi, L., Weemstra, C., Verbeke, J., Ekstrom, G., Zunino, A. &
    Giardini, D. (2013). "On measuring surface wave phase velocity from
    station-station cross-correlation of ambient signal".
    Geophys. J. Int. 192, 346-358.  doi:10.1093/gji/ggs023

* **AKI** (seccion 3.2, ecs. 48-51) -- dominio de la frecuencia.  La parte
  real del espectro cruzado promediado es proporcional a ``J0(w*dx/c)``.  Sus
  cruces por cero dan ``c(w_i) = w_i*dx / z_n``, con ``z_n`` los ceros de la
  funcion de Bessel J0.  **No requiere campo lejano**, asi que es el metodo
  adecuado cuando ``dx ~ lambda`` -- que es exactamente el regimen de este
  pipeline (``FAR_FIELD_FACTOR = 1.0`` en ``main_invert.py``).  Es el metodo
  principal de este modulo: :func:`measure_phase_velocity_aki`.

* **FRY** (seccion 3.1) -- dominio del tiempo.  Filtro gaussiano de banda
  estrecha + ventana temporal centrada en la llegada de grupo; se lee la fase
  del espectro y se le suma ``pi/4``.  Ese ``pi/4`` sale de la aproximacion de
  campo lejano (ecs. 33-34, ``w*dx/c >> 1``), asi que aqui es **secundario**:
  sirve de control cruzado en los pares mas largos, no como medida principal.
  Ver :func:`measure_phase_velocity_fry`.


Cual usar: AKI
--------------
Medido sobre el sintetico exacto (``Re[espectro] = S*J0(w*dx/c)``, sin ruido)
en ``test_phase_velocity.py``, error mediano en ``c`` por banda de ``dx/lambda``
-- con lambda la longitud de onda de FASE, ``c*T``:

    dx/lambda      AKI        FRY      sesgo FRY
    ---------------------------------------------
    0.2 - 0.5     0.03 %    10.75 %     +10.8 %
    0.5 - 1.0     0.00 %     0.81 %      +0.1 %
    1.0 - 1.5     0.00 %     0.35 %      +0.1 %
    1.5 - 2.5     0.00 %     0.28 %      -0.1 %
    2.5 - 4.0     0.00 %     0.22 %      -0.2 %
    4.0 - 8.0     0.00 %     0.19 %      -0.1 %
    8.0 -  20     0.00 %     0.10 %      -0.1 %
     20 - 200     0.01 %     0.03 %      -0.0 %

AKI es exacto en todo el rango porque no aproxima nada: los ceros de ``J0`` son
los ceros de ``J0``.  FRY se degrada segun baja ``dx/lambda``, y por debajo de
~0.5 en toda la banda directamente no devuelve curva (ninguna rama ``2*pi*N``
resulta fisicamente admisible).  Es la misma tendencia que reportan Boschi et
al. en el pie de su Fig. 6 -- *"At low frequencies, and particularly at shorter
epicentral distances, the match is less accurate"* -- solo que aqui cuantificada.

Para este pipeline (T = 0.1-5 s, array denso) el extremo de periodo largo de la
banda es el que mas profundidad muestrea, y es justo donde ``dx/lambda`` es mas
pequeno: es decir, donde FRY es peor.  Por eso **AKI es el metodo de produccion
y FRY solo control cruzado en los pares con dx/lambda > 3**.

El unico punto a favor de FRY es que da una curva continua, mientras que AKI
solo mide en las frecuencias de los cruces por cero (pocas en pares cortos).
Para invertir, puntos discretos con exactitud conocida valen mas que una curva
densa y sesgada.


Que aporta -- y que NO aporta -- la curva de velocidad de grupo
---------------------------------------------------------------
Conviene tenerlo claro antes de interpretar los resultados, porque es la
limitacion de fondo de todo este problema.

Ambos metodos miden el numero de onda ``k(w) = w/c(w)`` **salvo un entero**:
en AKI no se sabe a que cero de Bessel ``z_n`` corresponde el primer cruce; en
FRY queda un ``2*pi*N`` libre en la fase.  La curva de grupo restringe
``dw/dk``, no ``k``:

* En AKI, sobre una rama ``m``, ``k_i = z_{m+i}/dx`` **exactamente**, de modo
  que ``U = dw/dk = 2*pi*(f_{i+1}-f_i)*dx / (z_{m+i+1}-z_{m+i})``.  Como los
  ceros de J0 estan separados por ``~pi`` (de 3.115 entre z1 y z2 a 3.1416
  asintoticamente), ese ``U`` predicho varia menos del 1% entre ramas.

  Consecuencia: comparar ``U`` predicho con ``U`` medido **valida la
  secuencia de cruces** (detecta cruces perdidos o espurios, que cambian el
  ``U`` predicho en un factor ~2), pero **no elige la rama**.

* Lo que si elige la rama es el nivel ABSOLUTO: ``c`` debe caer en un rango
  fisico y la razon ``U/c`` debe ser plausible (para el modo fundamental de
  Rayleigh en un medio con dispersion normal, ``c > U`` siempre, y
  tipicamente ``U/c`` entre ~0.45 y 1).  Como las ramas separan ``c`` por
  factores ``z_{m+i+1}/z_{m+i}`` que son grandes a indice bajo (z1->z2 es un
  factor 2.3), esa criba es potente justo en el regimen de distancias cortas
  de este pipeline, donde solo hay unos pocos cruces y de indice bajo.

Por eso :func:`select_branch` aplica filtros duros (rango de ``c``, dispersion
normal, razon ``U/c``) y **devuelve todas las ramas supervivientes**,
marcando ``ambiguous=True`` si queda mas de una, en vez de fingir una eleccion
unica.  Es el mismo planteamiento de la Fig. 3(c) / Fig. 6 del articulo, donde
se dibujan todas las ramas y se selecciona una.

Dos maneras de romper el empate cuando queda:

1. Pasar ``c_ref=(f_ref, c_ref)`` -- una curva de fase de referencia de un
   modelo 1-D previo.  Es lo que hacen Verbeke et al. con PREM (seccion 3.1),
   solo que PREM no sirve a 0.1-5 s: usa tu propio modelo local en cuanto
   tengas una primera inversion.
2. Coherencia entre pares: la rama correcta produce una ``c(T)`` que varia
   suavemente con la distancia interestacion en todo el array, mientras que un
   error de rama da saltos discretos.  Eso se resuelve fuera de este modulo,
   comparando los ficheros de salida de muchos pares.


Flujo recomendado: DOS PASADAS
-------------------------------
La seleccion de rama es el unico riesgo real del metodo, y esta medida.  En un
barrido sintetico de 184 casos (distancia, anchura de banda, ruido, c_min):

* pares que llegan a **una sola rama admisible**: aciertan el **100%** de las
  veces, con un error mediano del 0.14%;
* pares con **mas de una**: aciertan solo ~40%.  El ranking entre ramas
  admisibles es poco mejor que una moneda al aire, y un margen de puntuacion
  tampoco las separa (se probo: ningun umbral discrimina).

Por eso ``require_unique=True`` es el valor por defecto: escribir los pares
ambiguos envenenaria una inversion tomografica en silencio.

Lo que decide cuantos pares llegan a rama unica es la **anchura de la banda**:

    banda de la curva de grupo   factor en f   % con rama unica
    ----------------------------------------------------------
    T = 0.1 - 0.4 s                    4              0 %
    T = 0.1 - 0.8 s                    8              0 %
    T = 0.1 - 1.5 s                   15             27 %
    T = 0.1 - 3.0 s                   30             72 %
    T = 0.1 - 5.0 s                   50             77 %

De ahi el flujo en dos pasadas:

1. **Banda lo mas ancha posible.**  Pasa a AKI la curva de grupo SIN el filtro
   de campo lejano (``vg_i * t_i <= distance`` en ``main_invert.py``): AKI no
   necesita campo lejano para nada, y ese filtro solo te estrecha la banda,
   que es justo lo que arruina la seleccion de rama.
2. **Primera pasada** con ``require_unique=True`` (por defecto).  Los pares que
   devuelvan curva tienen la rama bien.
3. **Curva de referencia**: ``build_reference_curve(resultados)`` apila esos
   pares en una ``c_ref`` mediana.
4. **Segunda pasada** sobre los pares que fallaron, pasando ``c_ref=``.  En el
   test recupero 152 de 152 ambiguos, todos correctos.


Uso tipico dentro de main_invert.py
-----------------------------------
Justo despues del bloque que escribe la curva de grupo de ``G_LR0``::

    from phase_velocity import measure_phase_velocity_aki, save_dispersion

    res = measure_phase_velocity_aki(
        G_LR0, dt=delta, dist_km=distance,
        T_vg=np.asarray(tfilt_), vg=np.asarray(vgfilt_),
        max_lag_s=4.0 * distance / FTAN_PARAMS['vg_min'],
    )
    if res['ok']:
        save_dispersion(f'{OUTPUT_PATH}/disp_phase_{sta1}_{sta2}_td.dat',
                        res['T'], res['c'], distance)

``G_LR0`` debe ser la traza YA plegada por ``symmetrize()`` (indice 0 = lag
cero).  Eso es justo lo que hace falta: si ``y`` es esa traza plegada,
``Re[rfft(y)]`` es exactamente el espectro (real) de la correlacion simetrica
de dos lados, que es la cantidad que aparece en la ec. (48).

Nota sobre convenciones de signo -- IMPORTANTE
----------------------------------------------
Los ceros de ``J0`` no se mueven si la correlacion cambia de signo global, asi
que la convencion de signo ZZ/RR no afecta a AKI.  Lo que SI afecta, y mucho,
es un error de 90 grados -- que es justo lo que se manipula en la separacion de
modos (``np.exp(+-1j*np.pi/2)`` sobre RZ/ZR en ``main_invert.py``).  Si esos
desfases estan bien puestos, ``G_LR0`` queda en fase con ``ZZ`` y la parte real
del espectro es tipo ``J0``; si estan cruzados, sale una mezcla con ``Y0`` y
los cruces se desplazan hasta medio lobulo.  En los tests sinteticos ese error
llega a falsear ``c`` un 65%.

Y **no se puede detectar desde un solo par**: un error de cuadratura suma una
constante a la fase, igual que un error de rama, asi que desplaza los cruces en
bloque sin tocar sus espaciados -- ni el ``U`` implicito ni la calidad del
ajuste a ``J0`` se enteran (:func:`bessel_fit_quality` documenta por que es
circular).  Hay que descartarlo aparte, una sola vez para todo el proyecto:

    from phase_velocity import crossings_of, quadrature_offset

    kw = dict(dt=delta, dist_km=distance, T_vg=T_pk_ZZ, vg=vg_pk_ZZ,
              max_lag_s=6*distance/0.5)
    print(quadrature_offset(crossings_of(ZZ, **kw), crossings_of(G_LR0, **kw)))

sobre un par con buena SNR y en la banda donde ``ZZ`` esta dominado por el modo
fundamental.  ``offset ~ 0`` confirma la convencion; ``offset ~ +-0.5`` dice que
hay que intercambiar ``RZ`` y ``ZR`` (o cambiar el signo de los desfases) en
``main_invert.py`` antes de fiarse de ninguna velocidad de fase.
"""

from __future__ import annotations

import numpy as np
from scipy.special import jn_zeros, j0
from scipy.interpolate import LSQUnivariateSpline, interp1d
from scipy.optimize import brentq
from scipy.signal import hilbert


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

#: Ceros de la funcion de Bessel J0 (``z_n`` de la ec. 50).  Un par largo con
#: banda ancha puede dar cientos de cruces, asi que la tabla se hace generosa;
#: :func:`_zero_spacings` extiende ademas con el limite asintotico ``pi``.
J0_ZEROS = jn_zeros(0, 2000)


def _zero_spacings(n):
    """Spacings ``z_{i+1} - z_i`` of the first ``n`` zeros of J0.

    Falls back to the asymptotic value ``pi`` beyond the tabulated zeros -- the
    spacing is within 1% of ``pi`` from the very first gap, so the fallback is
    harmless and keeps very long, broadband pairs from overrunning the table.
    """
    n_tab = J0_ZEROS.size
    if n <= n_tab:
        return np.diff(J0_ZEROS[:n])
    return np.concatenate([np.diff(J0_ZEROS), np.full(n - n_tab, np.pi)])

#: Rango fisico por defecto de la velocidad de fase (km/s).  Deliberadamente
#: ancho; conviene estrecharlo con lo que se sepa del emplazamiento.
DEFAULT_C_MIN = 0.15
DEFAULT_C_MAX = 5.0

#: Rango admisible de la razon U/c (velocidad de grupo / velocidad de fase)
#: para el modo fundamental de Rayleigh con dispersion normal.  El limite
#: superior es 1 (c > U siempre); el inferior cubre gradientes fuertes.
DEFAULT_RATIO_MIN = 0.45
DEFAULT_RATIO_MAX = 1.02

#: Valor de U/c hacia el que se desempata cuando sobreviven varias ramas.
DEFAULT_RATIO_TARGET = 0.85

#: Desajuste relativo maximo tolerado entre el U predicho por la secuencia de
#: cruces y el U medido con FTAN.  Sirve para detectar cruces perdidos o
#: espurios (que suelen dar factores ~2), no para elegir rama.
DEFAULT_GROUP_TOL = 0.40


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def group_curve_to_frequency(T_vg, vg):
    """Turn a period-sampled group-velocity curve into a function of frequency.

    Parameters
    ----------
    T_vg, vg : array_like
        Period (s) and group velocity (km/s), in any order (they are sorted
        internally).  NaNs and non-positive entries are dropped.

    Returns
    -------
    (callable, float, float)
        ``(vg_of_f, fmin, fmax)`` where ``vg_of_f(f)`` linearly interpolates the
        measured group velocity and returns ``nan`` outside ``[fmin, fmax]``.
    """
    T_vg = np.asarray(T_vg, dtype=float)
    vg = np.asarray(vg, dtype=float)
    good = np.isfinite(T_vg) & np.isfinite(vg) & (T_vg > 0) & (vg > 0)
    T_vg, vg = T_vg[good], vg[good]
    if T_vg.size < 2:
        raise ValueError('the group-velocity curve needs at least 2 valid points')

    f = 1.0 / T_vg
    order = np.argsort(f)
    f, v = f[order], vg[order]
    # Periodos duplicados -> el interpolador se queja; se promedian.
    f, idx = np.unique(f, return_inverse=True)
    v = np.bincount(idx, weights=v) / np.bincount(idx)

    vg_of_f = interp1d(f, v, kind='linear', bounds_error=False, fill_value=np.nan)
    return vg_of_f, float(f[0]), float(f[-1])


def _tukey_tail(n, frac):
    """Half-Tukey taper: flat at index 0, cosine roll-off over the last ``frac``."""
    w = np.ones(n)
    ntap = int(round(frac * n))
    if ntap >= 2:
        ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, ntap)))
        w[n - ntap:] = ramp
    return w


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1: ESPECTRO REAL DE LA CORRELACION SIMETRICA
# ══════════════════════════════════════════════════════════════════════════════

def real_spectrum(folded, dt, max_lag_s=None, taper_frac=0.25, pad_factor=1):
    r"""Real part of the spectrum of a symmetric cross-correlation.

    If ``y`` is the folded (one-sided) correlation produced by
    ``main_invert.symmetrize`` -- ``y[0]`` = zero lag, ``y[k] = x[+k] + x[-k]``
    -- then

    .. math:: \Re\{\mathrm{rfft}(y)\}(\omega)
              = y_0 + \sum_{k>0} y_k \cos(\omega k)
              = \sum_{k=-M}^{M} x_k e^{-i\omega k}

    which is exactly the (real) spectrum of the two-sided symmetric
    correlation, i.e. the quantity proportional to :math:`J_0(\omega\Delta x/c)`
    in eq. (48) of Boschi et al. (2013).  No extra factor is needed.

    Parameters
    ----------
    folded : array_like
        One-sided correlation, zero lag at index 0.
    dt : float
        Sample interval (s).
    max_lag_s : float, optional
        Truncate the correlation at this lag before transforming.  Strongly
        recommended: the stacked correlation is typically hundreds of seconds
        long while the surface-wave train lasts a few seconds, so keeping the
        whole trace buries the :math:`J_0` oscillation in coda noise.  A sane
        starting value is a few times the slowest expected arrival,
        ``4 * dist_km / vg_min``.  The trade-off is spectral resolution: the
        truncation smears the spectrum by ``~1/max_lag_s``, which must stay
        well below the lobe spacing ``c/(2*dist_km)`` -- :func:`spectral_resolution_check`
        quantifies it.  ``None`` (default) keeps the full trace.
    taper_frac : float, optional
        Fraction of the retained window given a cosine roll-off at the far
        (large-lag) end.  The zero-lag end is never tapered.  Default 0.25.
    pad_factor : int, optional
        Zero-pad the tapered window to ``pad_factor`` times the ORIGINAL trace
        length before transforming.  Padding adds no information but
        interpolates the spectrum, which makes the zero-crossing search much
        better conditioned after a short truncation.  Default 1.

    Returns
    -------
    dict
        ``{'freqs', 'real', 'window_length_s', 'n_fft'}``.
    """
    y = np.asarray(folded, dtype=float).copy()
    if y.ndim != 1 or y.size < 8:
        raise ValueError('folded must be a 1-D array with at least 8 samples')
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    n_full = y.size
    if max_lag_s is not None:
        n_keep = int(round(float(max_lag_s) / dt)) + 1
        n_keep = int(np.clip(n_keep, 8, n_full))
        y = y[:n_keep] * _tukey_tail(n_keep, taper_frac)

    window_length_s = y.size * dt
    n_fft = max(int(pad_factor) * n_full, y.size)

    spec = np.fft.rfft(y, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=dt)

    return {'freqs': freqs, 'real': spec.real,
            'window_length_s': window_length_s, 'n_fft': n_fft}


def spectral_resolution_check(window_length_s, dist_km, c_typical):
    """Compare the spectral smearing of the lag window with the J0 lobe spacing.

    Consecutive zeros of :math:`J_0` are ``~pi`` apart in ``w*dx/c``, i.e.
    ``df_lobe ~= c / (2*dist_km)`` in frequency.  A lag window of length ``L``
    smears the spectrum by ``df_win ~= 1/L``.  The zero crossings are only
    trustworthy while ``df_win`` is a small fraction of ``df_lobe``.

    Empirically (synthetic tests in ``test_phase_velocity.py``) the crossings
    stay accurate to better than ~1% up to ``ratio ~ 1.2`` -- far more tolerant
    than amplitude-based measurements, because smearing a symmetric oscillation
    displaces its zeros only at second order.  Past ``ratio ~ 1.5`` lobes start
    being lost and the error grows fast.  The sweet spot is ``ratio ~ 0.3-0.6``:
    short enough to reject coda noise, long enough to resolve the lobes.

    Returns
    -------
    dict
        ``{'df_window', 'df_lobe', 'ratio', 'ok'}`` with ``ok`` True when
        ``ratio = df_window / df_lobe < 1.0``.
    """
    df_window = 1.0 / float(window_length_s)
    df_lobe = float(c_typical) / (2.0 * float(dist_km))
    ratio = df_window / df_lobe if df_lobe > 0 else np.inf
    return {'df_window': df_window, 'df_lobe': df_lobe,
            'ratio': ratio, 'ok': bool(ratio < 1.0)}


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2: SUAVIZADO CON SPLINES Y CRUCES POR CERO
# ══════════════════════════════════════════════════════════════════════════════

def smooth_real_spectrum(freqs, re, fmin, fmax, dist_km, c_min,
                         knots_per_lobe=8.0):
    """Least-squares cubic-spline fit of the real cross-spectrum.

    Follows section 3.2 of Boschi et al. (2013): *"we determine the linear
    combination of cubic splines that best fits (in least-squares sense) the
    observed coherency. Splines are equally spaced, and spacing must be
    selected so that 'splined' coherency is sufficiently smooth"*.  The knot
    spacing here is tied to the physics rather than guessed: the narrowest
    :math:`J_0` lobe in the band is ``c_min / (2*dist_km)`` wide, and we place
    ``knots_per_lobe`` knots across it.

    Parameters
    ----------
    freqs, re : numpy.ndarray
        Frequency axis (Hz) and real part of the cross-spectrum.
    fmin, fmax : float
        Analysis band (Hz).
    dist_km : float
        Inter-station distance (km).
    c_min : float
        Smallest phase velocity considered plausible (km/s); sets the finest
        lobe spacing that must be resolved.
    knots_per_lobe : float, optional
        Knots per J0 lobe.  Fewer -> smoother.  Default 8.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, callable, dict)
        ``(f_band, s_band, spline, info)`` -- the band-limited frequency axis,
        the smoothed spectrum on it, a callable evaluating the fit, and
        ``info`` with ``n_knots``, ``n_knots_wanted``, ``samples_per_lobe`` and
        ``knots_per_lobe_achieved``.  If the spline cannot be built (too few
        samples), the raw spectrum is returned with a linear interpolant.

    .. note::
       The knot count is capped at ``len(f_band)//6``, so for a very long pair
       in a slow medium the lobes can be denser in frequency than the spectrum
       can describe, and the fit silently smooths lobes away.
       ``knots_per_lobe_achieved`` reports it; a cubic spline still describes a
       lobe well down to ~3 knots per lobe, but below ~2.5 the crossings are not
       trustworthy and the correlation simply is not long enough to resolve
       ``c/(2*dist_km)``.
    """
    band = (freqs >= fmin) & (freqs <= fmax)
    f_band = freqs[band]
    r_band = re[band]
    if f_band.size < 16:
        raise ValueError(f'only {f_band.size} spectral samples in [{fmin}, {fmax}] Hz; '
                         'widen the band or reduce truncation of the correlation')

    df_lobe = float(c_min) / (2.0 * float(dist_km))
    knot_spacing = df_lobe / float(knots_per_lobe)
    n_knots = int(np.floor((f_band[-1] - f_band[0]) / knot_spacing)) - 1
    # Schoenberg-Whitney: hacen falta bastantes muestras por intervalo de nodo.
    n_knots = int(np.clip(n_knots, 0, f_band.size // 6))

    n_wanted = int(np.floor((f_band[-1] - f_band[0]) / knot_spacing)) - 1
    df_data = float(np.median(np.diff(f_band)))
    info = {'n_knots': n_knots, 'n_knots_wanted': max(n_wanted, 0),
            'samples_per_lobe': df_lobe / df_data if df_data > 0 else np.inf,
            'knots_per_lobe_achieved': (float(knots_per_lobe) * n_knots / n_wanted
                                        if n_wanted > 0 else float(knots_per_lobe))}

    if n_knots >= 1:
        knots = np.linspace(f_band[0], f_band[-1], n_knots + 2)[1:-1]
        try:
            spline = LSQUnivariateSpline(f_band, r_band, t=knots, k=3)
            return f_band, spline(f_band), spline, info
        except Exception:
            pass  # cae al interpolador crudo

    raw = interp1d(f_band, r_band, kind='linear',
                   bounds_error=False, fill_value='extrapolate')
    return f_band, r_band, raw, info


def find_zero_crossings(f_band, s_band, spline, lobe_frac=0.03, min_sep=None):
    """Locate the zero crossings of the smoothed real cross-spectrum.

    Sign changes are detected on the dense frequency grid and refined with
    Brent's method on the spline.  Crossings that only bound negligible lobes
    (pure noise wiggles about zero) are discarded.

    Parameters
    ----------
    f_band, s_band : numpy.ndarray
        Band-limited frequency axis and smoothed spectrum.
    spline : callable
        The smoothed-spectrum evaluator returned by :func:`smooth_real_spectrum`.
    lobe_frac : float, optional
        A crossing is kept only if at least one of its two adjacent lobes peaks
        above ``lobe_frac * max|s_band|``.  Default 0.03.  Set 0 to disable.
    min_sep : float, optional
        Minimum separation (Hz) between accepted crossings; removes numerical
        doublets.  Default: a twentieth of the median spacing found.

    Returns
    -------
    numpy.ndarray
        Crossing frequencies (Hz), ascending.
    """
    s = np.asarray(s_band, dtype=float)
    sign_change = np.where(np.signbit(s[:-1]) != np.signbit(s[1:]))[0]
    if sign_change.size == 0:
        return np.empty(0)

    roots = []
    for i in sign_change:
        a, b = f_band[i], f_band[i + 1]
        try:
            if spline(a) == 0.0:
                roots.append(float(a))
            elif spline(a) * spline(b) < 0:
                roots.append(float(brentq(lambda x: float(spline(x)), a, b)))
            else:  # el signo cambia entre muestras pero el spline no: lineal
                roots.append(float(a + (b - a) * abs(s[i]) / (abs(s[i]) + abs(s[i + 1]))))
        except Exception:
            continue
    roots = np.unique(np.asarray(roots, dtype=float))
    if roots.size == 0:
        return roots

    if min_sep is None:
        min_sep = 0.05 * np.median(np.diff(roots)) if roots.size > 1 else 0.0
    if min_sep > 0 and roots.size > 1:
        keep = [roots[0]]
        for r in roots[1:]:
            if r - keep[-1] >= min_sep:
                keep.append(r)
        roots = np.asarray(keep)

    # Criba por amplitud de lobulo: un cruce real separa dos lobulos con
    # amplitud apreciable.  Ruido alrededor de cero genera cruces con lobulos
    # minusculos a ambos lados.
    if lobe_frac > 0 and roots.size:
        ref = np.max(np.abs(s))
        edges = np.concatenate([[f_band[0]], roots, [f_band[-1]]])
        lobe_amp = np.empty(edges.size - 1)
        for j in range(edges.size - 1):
            seg = (f_band >= edges[j]) & (f_band <= edges[j + 1])
            lobe_amp[j] = np.max(np.abs(s[seg])) if seg.any() else 0.0
        good = np.array([max(lobe_amp[j], lobe_amp[j + 1]) >= lobe_frac * ref
                         for j in range(roots.size)])
        roots = roots[good]

    return roots


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3: RAMAS DE VELOCIDAD DE FASE Y SELECCION
# ══════════════════════════════════════════════════════════════════════════════

def branch_phase_velocity(f_cross, dist_km, m):
    """Phase velocity of branch ``m`` (eq. 51 of Boschi et al. 2013).

    The i-th zero crossing is assigned to the ``(m+i)``-th zero of :math:`J_0`,
    so ``c_i = 2*pi*f_i*dist_km / z_{m+i}``.

    Parameters
    ----------
    f_cross : numpy.ndarray
        Zero-crossing frequencies (Hz), ascending.
    dist_km : float
        Inter-station distance (km).
    m : int
        0-based branch offset: ``m = 0`` assigns the first crossing to
        ``z_1 = 2.4048``.

    Returns
    -------
    numpy.ndarray
        Phase velocities (km/s) at ``f_cross``.
    """
    z = J0_ZEROS[m:m + f_cross.size]
    if z.size != f_cross.size:
        raise ValueError('branch offset exceeds the tabulated J0 zeros')
    return 2.0 * np.pi * f_cross * float(dist_km) / z


def branch_group_velocity(f_cross, dist_km, m):
    r"""Group velocity implied by the crossing sequence of branch ``m``.

    On a branch the wavenumber is known exactly at each crossing,
    ``k_i = z_{m+i} / dx``, so

    .. math:: U_{i+1/2} = \frac{d\omega}{dk}
              = \frac{2\pi (f_{i+1}-f_i)\,\Delta x}{z_{m+i+1}-z_{m+i}}

    evaluated at the midpoint frequency.  Because consecutive zeros of
    :math:`J_0` are separated by ``~pi`` whatever their index, this is almost
    independent of ``m`` -- it validates the crossing SEQUENCE (a missed or
    spurious crossing shows up as a factor ~2 error) rather than selecting the
    branch.  See the module docstring.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(f_mid, U)`` -- midpoint frequencies (Hz) and group velocities (km/s).
        Empty if fewer than two crossings.
    """
    if f_cross.size < 2:
        return np.empty(0), np.empty(0)
    z = J0_ZEROS[m:m + f_cross.size]
    f_mid = 0.5 * (f_cross[1:] + f_cross[:-1])
    U = 2.0 * np.pi * np.diff(f_cross) * float(dist_km) / np.diff(z)
    return f_mid, U


def trim_crossing_sequence(f_cross, dist_km, vg_of_f, tol=DEFAULT_GROUP_TOL):
    r"""Drop zero crossings that break the consecutive-``z_n`` assumption.

    Every method here assumes the detected crossings are CONSECUTIVE zeros of
    :math:`J_0`.  A spurious crossing (noise wiggling about zero) or a missed
    one (a lobe buried in noise) breaks that, and everything downstream of the
    break is wrong by a whole branch step.

    This is exactly what the measured group-velocity curve is good for.  The
    group velocity implied by a pair of consecutive crossings,

    .. math:: U = \frac{2\pi (f_{i+1}-f_i)\,\Delta x}{z_{n+1}-z_n}
              \approx 2 (f_{i+1}-f_i)\,\Delta x ,

    is branch-independent to better than 1% (consecutive zeros of :math:`J_0`
    are ``~pi`` apart whatever their index).  So it can be compared with the
    FTAN group velocity *before* the branch is known: a spurious crossing
    halves the spacing and so halves the implied ``U``, a missed one doubles
    it.  Both are far outside ``tol``.

    The longest contiguous run of intervals that agree with the measured group
    velocity is kept, which in practice trims the noisy band edges -- the same
    "remove, don't smooth" philosophy as ``robust_dispersion_clean.py``.

    Parameters
    ----------
    f_cross : numpy.ndarray
        Crossing frequencies (Hz), ascending.
    dist_km : float
        Inter-station distance (km).
    vg_of_f : callable
        Measured group velocity vs frequency.
    tol : float, optional
        Maximum relative deviation ``|U_implied/U_measured - 1|`` accepted for
        an interval.  Default :data:`DEFAULT_GROUP_TOL`.

    Returns
    -------
    (numpy.ndarray, int)
        ``(f_kept, n_dropped)``.
    """
    f_cross = np.asarray(f_cross, dtype=float)
    n = f_cross.size
    if n < 3:
        return f_cross, 0

    dz = _zero_spacings(n)
    f_mid = 0.5 * (f_cross[1:] + f_cross[:-1])
    U_imp = 2.0 * np.pi * np.diff(f_cross) * float(dist_km) / dz
    U_meas = np.asarray(vg_of_f(f_mid), dtype=float)

    with np.errstate(invalid='ignore', divide='ignore'):
        good = np.isfinite(U_meas) & (np.abs(U_imp / U_meas - 1.0) <= tol)

    if good.all():
        return f_cross, 0

    # Racha contigua mas larga de intervalos validos -> cruces [lo, hi]
    best_len, best = 0, (0, 0)
    i = 0
    while i < good.size:
        if good[i]:
            j = i
            while j + 1 < good.size and good[j + 1]:
                j += 1
            if (j - i + 1) > best_len:
                best_len, best = j - i + 1, (i, j)
            i = j + 1
        else:
            i += 1

    if best_len == 0:
        return np.empty(0), n
    lo, hi = best
    kept = f_cross[lo:hi + 2]
    return kept, n - kept.size


def trim_edge_outliers(f, c, tol=0.05):
    """Drop first/last points whose phase velocity breaks the curve's own trend.

    The outermost crossings sit where the J0 lobe that defines them is only
    partly inside the analysis band, so the spline's root there is biased and the
    point can be badly off while every interior point is good.  In an end-to-end
    synthetic pipeline run, every single point with >3% error (5 of 159) was the
    first or last of its curve.

    Each endpoint is compared with a linear extrapolation in ``log f`` from its
    two inner neighbours, and dropped if it deviates by more than ``tol``
    (relative).  Applied iteratively from each end.

    Parameters
    ----------
    f, c : numpy.ndarray
        Frequencies (Hz) and phase velocities (km/s), any consistent order.
    tol : float, optional
        Relative deviation above which an endpoint is dropped.  Default 0.05.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, int)
        ``(f_kept, c_kept, n_dropped)``.
    """
    f = np.asarray(f, dtype=float)
    c = np.asarray(c, dtype=float)
    order = np.argsort(f)
    f, c = f[order], c[order]
    n0 = f.size
    if n0 < 4:
        return f, c, 0

    changed = True
    while changed and f.size >= 4:
        changed = False
        for side in (0, -1):
            if f.size < 4:
                break
            if side == 0:
                x0, x1, x2 = np.log(f[0]), np.log(f[1]), np.log(f[2])
                y1, y2 = c[1], c[2]
            else:
                x0, x1, x2 = np.log(f[-1]), np.log(f[-2]), np.log(f[-3])
                y1, y2 = c[-2], c[-3]
            if x2 == x1:
                continue
            pred = y1 + (y2 - y1) * (x0 - x1) / (x2 - x1)
            if pred > 0 and abs(c[side] - pred) / pred > tol:
                f = f[1:] if side == 0 else f[:-1]
                c = c[1:] if side == 0 else c[:-1]
                changed = True

    return f, c, n0 - f.size


def bessel_fit_quality(f_band, s_band, f_cross, c_cross, dist_km):
    r"""Correlation between the observed real spectrum and :math:`J_0(\omega\Delta x/c)`.

    Measures how cleanly the observed spectrum oscillates like a Bessel
    function once ``c`` is known: it drops when the spectrum is noisy,
    incoherent, or contains lobes the crossing detector missed.  Use it as a
    per-pair quality weight.

    .. warning::
       This does **not** detect a quadrature (90-degree) error, and it cannot.
       The check is circular by construction -- ``c_cross`` was derived from the
       very crossings the model is being asked to reproduce, so the modelled
       :math:`J_0` has its zeros at the observed ones whatever the convention.
       Synthetic tests confirm it: mixing in a :math:`Y_0` component up to a
       full 90 degrees leaves this correlation at ~0.97 while the phase velocity
       drifts by up to 65%.

       More fundamentally, a quadrature error adds a constant to the phase, and
       so is indistinguishable from a branch error using this pair alone --
       both shift the crossings bodily, leaving their spacings (and hence the
       implied group velocity) untouched.  Use :func:`quadrature_offset` against
       a reference trace, or an external reference curve, to rule it out.

    Returns
    -------
    float
        Pearson correlation in ``[-1, 1]`` over the band spanned by the
        crossings, or ``nan``.  The absolute value is what matters (an overall
        sign flip is harmless).
    """
    if f_cross.size < 2:
        return np.nan
    c_of_f = interp1d(f_cross, c_cross, kind='linear',
                      bounds_error=False, fill_value=(c_cross[0], c_cross[-1]))
    seg = (f_band >= f_cross[0]) & (f_band <= f_cross[-1])
    if seg.sum() < 8:
        return np.nan
    model = j0(2.0 * np.pi * f_band[seg] * float(dist_km) / c_of_f(f_band[seg]))
    obs = s_band[seg]
    if np.std(obs) == 0 or np.std(model) == 0:
        return np.nan
    return float(np.corrcoef(obs, model)[0, 1])


def quadrature_offset(f_cross_ref, f_cross_test):
    r"""Fractional-lobe offset between two traces' zero-crossing sets.

    The non-circular way to check the ``+-90`` degree convention of the mode
    separation in ``main_invert.py``.  Compare a reference trace whose phase
    convention is not in question -- ``ZZ``, which needs no Hilbert-type
    manipulation -- with the synthesized ``G_LR0``, over a band where ``ZZ`` is
    dominated by the fundamental mode.  Both see the same medium, so their
    crossings must fall in the same places:

    * offset ``~0``      -> same convention, the ``exp(+-1j*pi/2)`` factors are right;
    * offset ``~+-0.5``  -> quadrature error, the crossings sit half a lobe off
      and every phase velocity is wrong by roughly half a branch step;
    * offset ``~+-1``    -> a whole-branch discrepancy (usually a lobe lost to
      noise in one of the two traces, not a convention problem).

    Parameters
    ----------
    f_cross_ref, f_cross_test : array_like
        Ascending crossing frequencies (Hz) of the reference and test traces.

    Returns
    -------
    dict
        ``{'offset', 'scatter', 'n', 'verdict'}``.  ``offset`` is the median
        signed offset in lobes, wrapped to ``[-0.5, 0.5]``; ``scatter`` its
        median absolute deviation; ``n`` the number of test crossings compared.
        ``offset``/``verdict`` are ``nan``/``'undetermined'`` if the two sets do
        not overlap in at least 3 crossings.
    """
    a = np.sort(np.asarray(f_cross_ref, dtype=float))
    b = np.sort(np.asarray(f_cross_test, dtype=float))
    if a.size < 2 or b.size < 1:
        return {'offset': np.nan, 'scatter': np.nan, 'n': 0, 'verdict': 'undetermined'}

    inside = (b >= a[0]) & (b <= a[-1])
    b = b[inside]
    if b.size < 3:
        return {'offset': np.nan, 'scatter': np.nan, 'n': int(b.size),
                'verdict': 'undetermined'}

    # Indice de lobulo de la referencia, interpolado a las frecuencias de test.
    idx = np.interp(b, a, np.arange(a.size, dtype=float))
    dev = idx - np.round(idx)
    dev = (dev + 0.5) % 1.0 - 0.5          # envuelto a [-0.5, 0.5)
    offset = float(np.median(dev))
    scatter = float(np.median(np.abs(dev - offset)))

    if scatter > 0.2:
        verdict = 'inconsistent (scatter too large to interpret)'
    elif abs(offset) < 0.15:
        verdict = 'consistent'
    elif abs(offset) > 0.35:
        verdict = 'QUADRATURE ERROR: crossings offset by ~half a lobe'
    else:
        verdict = 'marginal'
    return {'offset': offset, 'scatter': scatter, 'n': int(b.size), 'verdict': verdict}


def crossings_of(folded, dt, dist_km, T_vg, vg, **kw):
    """Convenience: just the zero-crossing frequencies of a trace.

    Same preprocessing as :func:`measure_phase_velocity_aki` (truncation,
    spline smoothing, lobe rejection) without the branch machinery.  Handy for
    feeding :func:`quadrature_offset` with a reference trace such as ``ZZ``.
    """
    kw.setdefault('trim_sequence', False)
    res = measure_phase_velocity_aki(folded, dt, dist_km, T_vg, vg, **kw)
    return res.get('f_cross_all', np.empty(0))


def select_branch(f_cross, dist_km, vg_of_f,
                  c_min=DEFAULT_C_MIN, c_max=DEFAULT_C_MAX,
                  ratio_min=DEFAULT_RATIO_MIN, ratio_max=DEFAULT_RATIO_MAX,
                  ratio_target=DEFAULT_RATIO_TARGET,
                  group_tol=DEFAULT_GROUP_TOL, max_branch=40,
                  c_ref=None):
    """Rank the candidate branches and pick one.

    Hard filters (a branch failing any of them is rejected outright):

    1. every ``c_i`` inside ``[c_min, c_max]``;
    2. normal dispersion, ``c_i > U_i`` at every crossing (allowing
       ``ratio_max`` slightly above 1 for measurement noise);
    3. median ``U/c`` inside ``[ratio_min, ratio_max]``;
    4. the group velocity implied by the crossing sequence agrees with the
       measured one to within ``group_tol`` (relative RMS) -- this catches
       missed or spurious crossings.

    Surviving branches are ranked by ``score``: the group-velocity misfit, plus
    a penalty on ``|log(median(U/c) / ratio_target)|``, plus a roughness
    penalty on ``c`` as a function of ``log f``.  If ``c_ref`` is given it
    replaces the ranking entirely with the RMS relative distance to that
    reference curve -- the criterion actually used by Verbeke et al. (PREM in
    their case; use your own local model here).

    Parameters
    ----------
    f_cross : numpy.ndarray
        Zero-crossing frequencies (Hz), ascending.
    dist_km : float
        Inter-station distance (km).
    vg_of_f : callable
        Measured group velocity as a function of frequency (see
        :func:`group_curve_to_frequency`).  ``nan`` outside the measured band.
    c_ref : (array_like, array_like), optional
        ``(f_ref, c_ref)`` reference phase-velocity curve.

    Returns
    -------
    dict
        ``{'best', 'candidates', 'ambiguous', 'n_survivors'}``.  ``best`` is the
        winning candidate dict (or ``None``), ``candidates`` the full table
        sorted by score, each entry carrying ``m``, ``c``, ``f_mid``,
        ``U_pred``, ``U_meas``, ``group_misfit``, ``ratio_median``, ``roughness``,
        ``score``, ``passed`` and ``reject_reason``.
    """
    f_cross = np.asarray(f_cross, dtype=float)
    if f_cross.size == 0:
        return {'best': None, 'candidates': [], 'ambiguous': False, 'n_survivors': 0}

    vg_at_cross = np.asarray(vg_of_f(f_cross), dtype=float)
    if not np.all(np.isfinite(vg_at_cross)):
        raise ValueError('some zero crossings fall outside the measured group-velocity '
                         'band; restrict the analysis band first')

    c_ref_fn = None
    if c_ref is not None:
        f_r = np.asarray(c_ref[0], dtype=float)
        c_r = np.asarray(c_ref[1], dtype=float)
        order = np.argsort(f_r)
        c_ref_fn = interp1d(f_r[order], c_r[order], kind='linear',
                            bounds_error=False,
                            fill_value=(c_r[order][0], c_r[order][-1]))

    n_max = min(int(max_branch), J0_ZEROS.size - f_cross.size)
    candidates = []
    for m in range(max(n_max, 0)):
        c = branch_phase_velocity(f_cross, dist_km, m)
        f_mid, U_pred = branch_group_velocity(f_cross, dist_km, m)
        U_meas = np.asarray(vg_of_f(f_mid), dtype=float) if f_mid.size else np.empty(0)

        ratio = vg_at_cross / c
        ratio_median = float(np.median(ratio))

        if f_mid.size and np.all(np.isfinite(U_meas)):
            group_misfit = float(np.sqrt(np.mean(((U_pred - U_meas) / U_meas) ** 2)))
        else:
            group_misfit = np.nan

        if c.size >= 3:
            # Curvatura relativa de c a lo largo de la secuencia de cruces: una
            # rama equivocada produce una curva con codos, la correcta no.
            d2 = np.diff(c, 2) / np.mean(c)
            roughness = float(np.sqrt(np.mean(d2 ** 2)))
        else:
            roughness = 0.0

        reason = ''
        if np.any(c < c_min) or np.any(c > c_max):
            reason = f'c outside [{c_min}, {c_max}] km/s'
        elif np.any(ratio > ratio_max):
            reason = 'c < U somewhere (inverse dispersion / branch too high)'
        elif not (ratio_min <= ratio_median <= ratio_max):
            reason = f'median U/c = {ratio_median:.2f} outside [{ratio_min}, {ratio_max}]'
        elif np.isfinite(group_misfit) and group_misfit > group_tol:
            reason = f'group-velocity misfit {group_misfit:.2f} > {group_tol}'

        passed = reason == ''

        if c_ref_fn is not None:
            score = float(np.sqrt(np.mean(((c - c_ref_fn(f_cross)) / c_ref_fn(f_cross)) ** 2)))
        else:
            gm = group_misfit if np.isfinite(group_misfit) else 0.0
            score = (gm
                     + abs(np.log(max(ratio_median, 1e-6) / ratio_target))
                     + 2.0 * roughness)

        candidates.append({
            'm': m, 'f': f_cross, 'c': c,
            'f_mid': f_mid, 'U_pred': U_pred, 'U_meas': U_meas,
            'group_misfit': group_misfit,
            'ratio': ratio, 'ratio_median': ratio_median,
            'roughness': roughness, 'score': score,
            'passed': passed, 'reject_reason': reason,
        })

        # Mas alla del punto en que c cae bajo c_min ya no hay nada que ganar.
        if np.all(c < c_min):
            break

    survivors = [cd for cd in candidates if cd['passed']]
    survivors.sort(key=lambda d: d['score'])
    candidates.sort(key=lambda d: (not d['passed'], d['score']))

    return {'best': survivors[0] if survivors else None,
            'candidates': candidates,
            'ambiguous': len(survivors) > 1,
            'n_survivors': len(survivors)}


# ══════════════════════════════════════════════════════════════════════════════
# METODO PRINCIPAL: AKI (seccion 3.2)
# ══════════════════════════════════════════════════════════════════════════════

def measure_phase_velocity_aki(folded, dt, dist_km, T_vg, vg,
                               fmin=None, fmax=None,
                               max_lag_s=None, taper_frac=0.25, pad_factor=1,
                               c_min=DEFAULT_C_MIN, c_max=DEFAULT_C_MAX,
                               knots_per_lobe=8.0, lobe_frac=0.03,
                               auto_pad=True, max_pad_factor=16,
                               trim_sequence=True, require_unique=True,
                               c_ref=None, verbose=False, **select_kw):
    """Phase velocity by Bessel-function zero crossings (AKI, section 3.2).

    The full chain: real spectrum of the symmetric correlation -> cubic-spline
    smoothing -> zero crossings -> candidate branches -> branch selection using
    the measured group-velocity curve.

    Parameters
    ----------
    folded : array_like
        Folded (one-sided) cross-correlation, zero lag at index 0 -- i.e. the
        output of ``main_invert.symmetrize``.  Use the same trace whose group
        velocity was measured: ``G_LR0`` for the fundamental mode, ``G_LR1``
        for the first higher mode, or ``ZZ`` if you are not separating modes.
    dt : float
        Sample interval (s).
    dist_km : float
        Inter-station distance (km).
    T_vg, vg : array_like
        The measured group-velocity dispersion curve (period in s, velocity in
        km/s) for this same trace -- e.g. ``tfilt_``, ``vgfilt_`` in
        ``main_invert.py``.
    fmin, fmax : float, optional
        Analysis band (Hz).  Default: the band actually covered by the
        group-velocity curve, which is also the band where branch selection is
        possible at all.
    max_lag_s, taper_frac, pad_factor : optional
        Passed to :func:`real_spectrum`.  Setting ``max_lag_s`` to a few times
        ``dist_km / vg_min`` is strongly recommended -- see that function.
    c_min, c_max : float, optional
        Plausible phase-velocity range (km/s).  ``c_min`` also sets the spline
        knot spacing, so do not make it absurdly small.
    knots_per_lobe, lobe_frac : optional
        Smoothing and crossing-acceptance controls.
    trim_sequence : bool, optional
        Drop crossings whose implied group velocity contradicts FTAN, keeping
        the longest self-consistent run (see :func:`trim_crossing_sequence`).
        This is what protects the measurement from spurious or missed lobes at
        the noisy band edges.  Default ``True``.
    auto_pad : bool, optional
        Raise ``pad_factor`` automatically when the spline's knot budget, not
        the data, is what limits lobe resolution (default ``True``).  Padding
        adds no information -- it interpolates a band-limited spectrum -- but it
        supplies the support points the least-squares spline needs.  Only
        applied when the unpadded data already carry >= 4 samples per lobe.
    max_pad_factor : int, optional
        Ceiling for ``auto_pad``.  Default 16.
    require_unique : bool, optional
        Refuse to return a curve when more than one branch is admissible
        (``ok`` stays ``False``).  **Default ``True``, and leaving it on is
        strongly advised.**  In a 184-case synthetic sweep, pairs that reached
        exactly one admissible branch were right 100% of the time (median error
        0.14%), while pairs with more than one were right only ~40% of the time
        -- the score-based ranking between admissible branches is close to a
        coin flip, and a score margin does not separate them either.  Writing
        those out would silently poison a tomographic inversion.  Ignored when
        ``c_ref`` is supplied, since the reference curve resolves the choice.
    c_ref : (array_like, array_like), optional
        Reference phase-velocity curve ``(f_ref, c_ref)`` used to break branch
        ambiguity, as PREM is used in section 3.1 of the paper.
    verbose : bool, optional
        Print a short report.
    **select_kw
        Forwarded to :func:`select_branch` (``ratio_min``, ``ratio_max``,
        ``ratio_target``, ``group_tol``, ``max_branch``).

    Returns
    -------
    dict
        ``ok`` says whether a curve was produced.  On success: ``T`` (period, s,
        ascending), ``c`` (phase velocity, km/s), ``f``, ``branch``,
        ``ambiguous``, ``n_survivors``, ``bessel_corr``, ``resolution``,
        ``candidates``, plus the spectrum arrays (``freqs``, ``real``,
        ``f_band``, ``s_band``, ``f_cross``) for plotting, and ``messages``.
    """
    messages = []
    out = {'ok': False, 'method': 'aki', 'dist_km': float(dist_km),
           'messages': messages}

    vg_of_f, f_lo, f_hi = group_curve_to_frequency(T_vg, vg)
    fmin = f_lo if fmin is None else max(float(fmin), f_lo)
    fmax = f_hi if fmax is None else min(float(fmax), f_hi)
    if not (fmax > fmin):
        messages.append('empty analysis band after intersecting with the group curve')
        return out
    out['band'] = (fmin, fmax)

    spec = real_spectrum(folded, dt, max_lag_s=max_lag_s,
                         taper_frac=taper_frac, pad_factor=pad_factor)

    # ── Presupuesto de nodos del spline ──────────────────────────────────────
    # smooth_real_spectrum limita los nodos a len(f_band)//6 por estabilidad
    # numerica del ajuste.  Si los lobulos son densos en frecuencia, ese tope
    # muerde ANTES que la resolucion real de los datos: el spline sobresuaviza,
    # se pierden lobulos, y el recuento de cruces -- y con el la rama -- salen
    # mal.  En un test sintetico eso da un 13% de error y una rama equivocada.
    # El zero-padding no anade informacion, pero interpola el espectro y le da
    # al spline los puntos de apoyo que necesita.  Se sube solo lo justo.
    df_lobe_min = float(c_min) / (2.0 * float(dist_km))
    n_wanted = max(int(np.floor((fmax - fmin) / (df_lobe_min / knots_per_lobe))) - 1, 0)
    n_band = int(((spec['freqs'] >= fmin) & (spec['freqs'] <= fmax)).sum())
    df_native = (float(np.median(np.diff(spec['freqs']))) * pad_factor) if n_band else np.inf
    samples_per_lobe_native = df_lobe_min / df_native if df_native > 0 else np.inf

    if auto_pad and n_band and n_band < 6 * n_wanted and samples_per_lobe_native >= 4:
        new_pad = min(int(max_pad_factor),
                      pad_factor * int(np.ceil(6 * n_wanted / n_band)))
        if new_pad > pad_factor:
            spec = real_spectrum(folded, dt, max_lag_s=max_lag_s,
                                 taper_frac=taper_frac, pad_factor=new_pad)
            messages.append(f'zero-padded {new_pad}x (from {pad_factor}x) so the spline '
                            f'can resolve the J0 lobes; the data themselves carry '
                            f'{samples_per_lobe_native:.0f} samples per lobe')
            pad_factor = new_pad
    out['pad_factor'] = pad_factor
    out['freqs'], out['real'] = spec['freqs'], spec['real']

    c_typ = float(np.nanmedian(vg_of_f(np.linspace(fmin, fmax, 32))))
    res = spectral_resolution_check(spec['window_length_s'], dist_km, c_typ)
    out['resolution'] = res
    if not res['ok']:
        messages.append(
            f"lag window {spec['window_length_s']:.1f} s smears the spectrum by "
            f"{res['df_window']:.4f} Hz vs. a J0 lobe spacing of {res['df_lobe']:.4f} Hz "
            f"(ratio {res['ratio']:.2f} > 1); lobes may be unresolved -- "
            'lengthen max_lag_s')

    try:
        f_band, s_band, spline, sm_info = smooth_real_spectrum(
            spec['freqs'], spec['real'], fmin, fmax, dist_km, c_min,
            knots_per_lobe=knots_per_lobe)
    except ValueError as exc:
        messages.append(str(exc))
        return out
    out['f_band'], out['s_band'] = f_band, s_band
    out['smoothing'] = sm_info

    if sm_info['samples_per_lobe'] < 6:
        messages.append(
            f"only {sm_info['samples_per_lobe']:.1f} spectral samples per J0 lobe "
            f"(lobe spacing {c_min/(2*dist_km):.4f} Hz): the correlation is too short "
            'to resolve the lobes of a pair this long -- crossings will be lost')
    elif sm_info['knots_per_lobe_achieved'] < 2.5:
        messages.append(
            f"RESULT UNRELIABLE: spline limited to {sm_info['n_knots']} of the "
            f"{sm_info['n_knots_wanted']} knots needed "
            f"({sm_info['knots_per_lobe_achieved']:.1f} per lobe), so lobes are being "
            'smoothed away and the branch is likely wrong. Raise c_min to the slowest '
            'phase velocity you can actually justify, or raise max_pad_factor')

    f_cross_all = find_zero_crossings(f_band, s_band, spline, lobe_frac=lobe_frac)
    out['f_cross_all'] = f_cross_all
    if f_cross_all.size == 0:
        messages.append('no zero crossing of Re[cross-spectrum] inside the band')
        return out

    if trim_sequence:
        f_cross, n_dropped = trim_crossing_sequence(
            f_cross_all, dist_km, vg_of_f,
            tol=select_kw.get('group_tol', DEFAULT_GROUP_TOL))
        if n_dropped:
            messages.append(f'{n_dropped} of {f_cross_all.size} crossings dropped: the '
                            'implied group velocity is inconsistent with FTAN there '
                            '(spurious or missed lobes, usually at the band edges)')
    else:
        f_cross, n_dropped = f_cross_all, 0
    out['f_cross'] = f_cross
    out['n_dropped'] = n_dropped

    if f_cross.size == 0:
        messages.append('no self-consistent run of zero crossings survived')
        return out
    if f_cross.size == 1:
        messages.append('only one zero crossing: a single phase-velocity point, and '
                        'the branch cannot be validated against the group curve')

    sel = select_branch(f_cross, dist_km, vg_of_f, c_min=c_min, c_max=c_max,
                        c_ref=c_ref, **select_kw)
    out['candidates'] = sel['candidates']
    out['ambiguous'] = sel['ambiguous']
    out['n_survivors'] = sel['n_survivors']

    best = sel['best']
    if best is None:
        reasons = {cd['reject_reason'] for cd in sel['candidates'][:6]}
        messages.append('no branch survived the physical filters: ' + '; '.join(sorted(reasons)))
        return out

    if sel['ambiguous']:
        alt = ', '.join(f"m={cd['m']} (score {cd['score']:.3f})"
                        for cd in sel['candidates'] if cd['passed'])
        messages.append(f"{sel['n_survivors']} branches are admissible [{alt}] -- "
                        'inspect plot_aki() or supply c_ref')
        if require_unique and c_ref is None:
            out['ambiguous'] = True
            return out

    corr = bessel_fit_quality(f_band, s_band, f_cross, best['c'], dist_km)
    out['bessel_corr'] = corr
    if np.isfinite(corr) and abs(corr) < 0.5:
        messages.append(f'low J0 fit quality (|corr| = {abs(corr):.2f}); check the '
                        '+-90 deg convention of the RZ/ZR mode separation')

    # f, T y c se devuelven alineados y en orden de PERIODO ASCENDENTE (=
    # frecuencia descendente), que es el orden en que se escriben los ficheros.
    # NOTA: se probo recortar aqui los puntos de borde con trim_edge_outliers.
    # Medido sobre la tirada sintetica completa del pipeline, EMPEORA el
    # resultado (mediana 0.16% -> 0.89%, puntos con >3% de error 5 -> 61): al
    # alterar las curvas cambia la c_ref que construye build_reference_curve, y
    # eso desestabiliza la seleccion de rama de la segunda pasada.  La funcion
    # queda disponible para aplicarla A POSTERIORI, por curva, si hace falta.
    order = np.argsort(f_cross)[::-1]
    out.update({'ok': True,
                'f': f_cross[order], 'T': 1.0 / f_cross[order], 'c': best['c'][order],
                'branch': best['m'],
                'U_meas': np.asarray(vg_of_f(f_cross))[order],
                'group_misfit': best['group_misfit'],
                'ratio_median': best['ratio_median']})

    if verbose:
        print(f"  AKI: {f_cross.size} crossings in [{fmin:.3f}, {fmax:.3f}] Hz, "
              f"branch m={best['m']}, U/c={best['ratio_median']:.2f}, "
              f"J0 corr={corr:.2f}, survivors={sel['n_survivors']}")
        for msg in messages:
            print(f'    ! {msg}')

    return out


# ══════════════════════════════════════════════════════════════════════════════
# METODO SECUNDARIO: FRY (seccion 3.1) -- control cruzado, requiere campo lejano
# ══════════════════════════════════════════════════════════════════════════════

def measure_phase_velocity_fry(folded, dt, dist_km, T_vg, vg,
                               fmin=None, fmax=None, df=None,
                               alpha=20.0, n_cycles=2.0,
                               window_center='group', quarter_sign=+1.0,
                               c_min=DEFAULT_C_MIN, c_max=DEFAULT_C_MAX,
                               c_ref=None, verbose=False, **select_kw):
    r"""Phase velocity in the time domain (FRY, section 3.1).

    For each frequency the folded correlation is narrowband filtered with a
    Gaussian, windowed in time around the group arrival ``dist/U(f)`` (the FTAN
    curve supplies that centre), and the phase of its spectrum is read.  Per
    eq. (35) and the remark in section 3.1, a quarter-cycle must be added:

    .. math:: \frac{\omega \Delta x}{c} = \phi_{obs} + \frac{\pi}{4} + 2\pi N

    .. warning::
       That ``pi/4`` is the **far-field** approximation (eqs. 33-34, valid for
       ``w*dx/c >> 1``).  With ``dist ~ lambda`` -- the regime of this pipeline
       -- it is a systematic bias, so treat FRY as a cross-check on the longest
       pairs only and trust :func:`measure_phase_velocity_aki` otherwise.  It is
       also specific to ambient-noise correlations: it must NOT be applied to
       ballistic earthquake signals (section 3.1).

    .. note::
       The sign convention here is ``phi = -angle(rfft(windowed))``, so a pure
       delay ``t0`` gives ``phi = +w*t0``.  Verify it once against AKI on a
       well-behaved long pair; ``quarter_sign=-1`` flips the ``pi/4``.

    Parameters
    ----------
    folded, dt, dist_km, T_vg, vg
        As in :func:`measure_phase_velocity_aki`.
    fmin, fmax : float, optional
        Analysis band (Hz).  Default: the band of the group-velocity curve.
    df : float, optional
        Frequency step (Hz).  Must be fine enough for the phase to advance by
        less than ``pi`` between samples, otherwise the unwrapping fails.
        Default: ``c_min / (4*dist_km)``.
    alpha : float, optional
        Gaussian bandpass width, ``exp(-alpha*((f-f0)/f0)**2)``.  Default 20
        (narrower than the FTAN step-1 filter).
    n_cycles : float, optional
        Gaussian time-window half-width in periods.  Default 2.
    window_center : {'group', 'max'}, optional
        Centre the time window on the FTAN group arrival (default) or on the
        envelope maximum of the filtered trace.
    quarter_sign : float, optional
        Sign of the ``pi/4`` correction.  Default ``+1``.

    Returns
    -------
    dict
        Same shape as :func:`measure_phase_velocity_aki`, with ``phi`` (the
        unwrapped phase) added and ``branch`` meaning the integer ``N``.
    """
    messages = []
    out = {'ok': False, 'method': 'fry', 'dist_km': float(dist_km),
           'messages': messages}

    vg_of_f, f_lo, f_hi = group_curve_to_frequency(T_vg, vg)
    fmin = f_lo if fmin is None else max(float(fmin), f_lo)
    fmax = f_hi if fmax is None else min(float(fmax), f_hi)
    if not (fmax > fmin):
        messages.append('empty analysis band after intersecting with the group curve')
        return out
    if df is None:
        df = c_min / (4.0 * float(dist_km))
    f_grid = np.arange(fmin, fmax + 0.5 * df, df)
    # np.arange puede rebasar fmax por redondeo, y ahi vg_of_f devuelve nan, lo
    # que envenena la mediana de U/c y hace que se rechacen TODAS las ramas.
    f_grid = f_grid[(f_grid >= fmin) & (f_grid <= fmax)]
    if f_grid.size >= 1:
        vg_chk = np.asarray(vg_of_f(f_grid), dtype=float)
        f_grid = f_grid[np.isfinite(vg_chk) & (vg_chk > 0)]
    if f_grid.size < 3:
        messages.append('analysis band too narrow for the required frequency step')
        return out
    out['band'] = (float(f_grid[0]), float(f_grid[-1]))

    y = np.asarray(folded, dtype=float)
    N = y.size
    t = np.arange(N) * dt
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(N, d=dt)

    phi_wrapped = np.empty(f_grid.size)
    for i, f0 in enumerate(f_grid):
        gauss = np.exp(-alpha * ((freqs - f0) / f0) ** 2)
        filt = np.fft.irfft(Y * gauss, n=N)

        if window_center == 'max':
            tau = t[int(np.argmax(np.abs(hilbert(filt))))]
        else:
            u = float(vg_of_f(f0))
            tau = float(dist_km) / u if np.isfinite(u) and u > 0 else t[N // 2]

        sigma = n_cycles / f0
        win = np.exp(-0.5 * ((t - tau) / sigma) ** 2)
        S = np.fft.rfft(filt * win)
        k0 = int(np.argmin(np.abs(freqs - f0)))
        phi_wrapped[i] = -np.angle(S[k0])

    phi = np.unwrap(phi_wrapped)
    # La fase de propagacion es positiva y creciente; se sube en bloques de 2pi
    # hasta que lo sea (el N global se elige despues).
    arg0 = phi + quarter_sign * np.pi / 4.0
    n_shift = int(np.ceil(max(0.0, -arg0.min()) / (2.0 * np.pi)))
    arg0 = arg0 + 2.0 * np.pi * n_shift
    out['phi'] = phi

    # Ramas: un unico entero N para toda la banda.
    n_lo = -int(np.floor(arg0.min() / (2.0 * np.pi)))
    candidates = []
    vg_grid = np.asarray(vg_of_f(f_grid), dtype=float)
    for Nn in range(n_lo, n_lo + 60):
        arg = arg0 + 2.0 * np.pi * Nn
        if np.any(arg <= 0):
            continue
        c = 2.0 * np.pi * f_grid * float(dist_km) / arg
        if np.all(c < c_min):
            break
        ratio = vg_grid / c
        ratio_median = float(np.nanmedian(ratio))
        # U implicito = dw/dk con k = arg/dx  ->  independiente de N, como en AKI.
        k = arg / float(dist_km)
        U_pred = np.gradient(2.0 * np.pi * f_grid, k)
        group_misfit = float(np.sqrt(np.nanmean(((U_pred - vg_grid) / vg_grid) ** 2)))

        reason = ''
        if np.any(c < c_min) or np.any(c > c_max):
            reason = f'c outside [{c_min}, {c_max}] km/s'
        elif not np.isfinite(ratio_median):
            reason = 'group velocity undefined over part of the band'
        elif np.any(ratio > select_kw.get('ratio_max', DEFAULT_RATIO_MAX)):
            reason = 'c < U somewhere'
        elif not (select_kw.get('ratio_min', DEFAULT_RATIO_MIN)
                  <= ratio_median <= select_kw.get('ratio_max', DEFAULT_RATIO_MAX)):
            reason = f'median U/c = {ratio_median:.2f} implausible'

        if c_ref is not None:
            f_r, c_r = np.asarray(c_ref[0], float), np.asarray(c_ref[1], float)
            o = np.argsort(f_r)
            ref = interp1d(f_r[o], c_r[o], bounds_error=False,
                           fill_value=(c_r[o][0], c_r[o][-1]))(f_grid)
            score = float(np.sqrt(np.mean(((c - ref) / ref) ** 2)))
        else:
            score = group_misfit + abs(np.log(max(ratio_median, 1e-6)
                                              / select_kw.get('ratio_target',
                                                              DEFAULT_RATIO_TARGET)))

        candidates.append({'m': Nn, 'f': f_grid, 'c': c, 'ratio_median': ratio_median,
                           'group_misfit': group_misfit, 'score': score,
                           'passed': reason == '', 'reject_reason': reason,
                           'f_mid': f_grid, 'U_pred': U_pred, 'U_meas': vg_grid,
                           'roughness': 0.0})

    survivors = sorted([cd for cd in candidates if cd['passed']], key=lambda d: d['score'])
    candidates.sort(key=lambda d: (not d['passed'], d['score']))
    out['candidates'] = candidates
    out['n_survivors'] = len(survivors)
    out['ambiguous'] = len(survivors) > 1

    if not survivors:
        messages.append('no admissible 2*pi*N branch')
        return out
    if out['ambiguous']:
        messages.append(f"{len(survivors)} admissible branches "
                        f"(N = {[cd['m'] for cd in survivors]}) -- supply c_ref")

    best = survivors[0]
    order = np.argsort(f_grid)[::-1]
    out.update({'ok': True, 'f': f_grid[order], 'T': 1.0 / f_grid[order],
                'c': best['c'][order], 'branch': best['m'],
                'group_misfit': best['group_misfit'],
                'ratio_median': best['ratio_median']})

    if verbose:
        print(f"  FRY: branch N={best['m']}, U/c={best['ratio_median']:.2f}, "
              f"survivors={len(survivors)}")
        for msg in messages:
            print(f'    ! {msg}')
    return out


# ══════════════════════════════════════════════════════════════════════════════
# COMPROBACION DE COHERENCIA CON LA CURVA DE GRUPO (via integracion)
# ══════════════════════════════════════════════════════════════════════════════

def phase_from_group_integration(T_vg, vg, f_anchor, c_anchor, f_out=None):
    r"""Phase velocity obtained by integrating the group-velocity curve.

    .. math:: k(\omega) = k(\omega_0) + \int_{\omega_0}^{\omega}
              \frac{d\omega'}{U(\omega')}, \qquad c(\omega) = \omega / k(\omega)

    This is the route suggested by Bensen et al. (2007) and explicitly judged
    insufficient by Boschi et al. (2013, section 3.1): it needs the integration
    constant ``c(f_anchor)``, so on its own it cannot identify phase velocity
    uniquely.  It IS useful once a direct measurement exists -- anchor it on one
    AKI crossing and check that the integrated curve reproduces the others.

    Parameters
    ----------
    T_vg, vg : array_like
        Measured group-velocity dispersion curve.
    f_anchor, c_anchor : float
        Frequency (Hz) and phase velocity (km/s) of the anchor point.
    f_out : array_like, optional
        Output frequencies (Hz).  Default: the group curve's own frequencies.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(f_out, c)`` with ``nan`` where the integrated wavenumber turns
        non-positive.
    """
    vg_of_f, f_lo, f_hi = group_curve_to_frequency(T_vg, vg)
    f_dense = np.linspace(f_lo, f_hi, 2001)
    u = np.asarray(vg_of_f(f_dense), dtype=float)
    w = 2.0 * np.pi * f_dense

    integ = np.concatenate([[0.0], np.cumsum(np.diff(w) * 0.5 * (1.0 / u[1:] + 1.0 / u[:-1]))])
    k_anchor = 2.0 * np.pi * float(f_anchor) / float(c_anchor)
    integ_anchor = float(np.interp(float(f_anchor), f_dense, integ))
    k = k_anchor + (integ - integ_anchor)

    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.where(k > 0, w / k, np.nan)

    if f_out is None:
        return f_dense, c
    f_out = np.asarray(f_out, dtype=float)
    return f_out, np.interp(f_out, f_dense, c, left=np.nan, right=np.nan)


# ══════════════════════════════════════════════════════════════════════════════
# SALIDA: FICHEROS Y FIGURAS
# ══════════════════════════════════════════════════════════════════════════════

def build_reference_curve(results, n_out=60, min_pairs=3):
    """Stack unambiguous per-pair results into a reference phase-velocity curve.

    The second half of the recommended two-pass workflow (see the module
    docstring).  Pass the ``measure_phase_velocity_aki`` results of the pairs
    that came back with a unique branch; the median curve they define is then
    handed back to the ambiguous pairs as ``c_ref``, which in synthetic tests
    recovered 152 of 152 of them.

    Parameters
    ----------
    results : iterable of dict
        Results from :func:`measure_phase_velocity_aki`.  Entries that are not
        ``ok``, or that are flagged ``ambiguous``, are ignored.
    n_out : int, optional
        Number of log-spaced output frequencies.  Default 60.
    min_pairs : int, optional
        Minimum number of contributing pairs required at a frequency for it to
        be kept.  Default 3.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray) or None
        ``(f_ref, c_ref)`` ready to pass as ``c_ref=``, or ``None`` if too few
        usable results were supplied.
    """
    good = [r for r in results
            if r.get('ok') and not r.get('ambiguous') and np.size(r.get('f', [])) >= 2]
    if len(good) < min_pairs:
        return None

    # Union de las bandas, no interseccion: con bandas heterogeneas la
    # interseccion se queda en nada.  El filtro n_contrib >= min_pairs de mas
    # abajo es el que recorta los extremos mal cubiertos.
    f_lo = min(float(np.min(r['f'])) for r in good)
    f_hi = max(float(np.max(r['f'])) for r in good)
    if not (f_hi > f_lo):
        return None

    f_out = np.logspace(np.log10(f_lo), np.log10(f_hi), int(n_out))
    stack = np.full((len(good), f_out.size), np.nan)
    for i, r in enumerate(good):
        order = np.argsort(r['f'])
        stack[i] = np.interp(f_out, np.asarray(r['f'])[order], np.asarray(r['c'])[order],
                             left=np.nan, right=np.nan)

    with np.errstate(all='ignore'):
        n_contrib = np.sum(np.isfinite(stack), axis=0)
        c_out = np.nanmedian(stack, axis=0)
    keep = (n_contrib >= min_pairs) & np.isfinite(c_out)
    if keep.sum() < 2:
        return None
    return f_out[keep], c_out[keep]


def save_dispersion(path, T, c, distance, comments=''):
    """Write a phase-velocity curve in the same format as the group-velocity files.

    Two columns (period in s ascending, velocity in km/s) with the inter-station
    distance as a bare header line -- identical to what ``main_invert.py``
    writes for ``disp_ZZ_*_td.dat``, so downstream tooling needs no change.
    """
    T = np.asarray(T, dtype=float)
    c = np.asarray(c, dtype=float)
    good = np.isfinite(T) & np.isfinite(c)
    T, c = T[good], c[good]
    order = np.argsort(T)
    header = f'{distance}' + (f'\n{comments}' if comments else '')
    np.savetxt(path, np.column_stack((T[order], c[order])), header=header, comments='')
    return path


def plot_aki(result, T_vg=None, vg=None, ax=None, xaxis='period', max_branches=12):
    """Diagnostic figure: real cross-spectrum, crossings, and every branch.

    Mirrors Figs. 3(c) and 6 of Boschi et al. (2013): all candidate branches as
    faint markers, the selected one highlighted, and the measured group-velocity
    curve for reference.  This is the figure to look at whenever ``ambiguous``
    is True.

    Parameters
    ----------
    result : dict
        Output of :func:`measure_phase_velocity_aki`.
    T_vg, vg : array_like, optional
        Measured group-velocity curve to overlay.
    ax : optional
        A pair of axes ``(ax_spec, ax_disp)``.  A new 2-panel figure is created
        if omitted.
    xaxis : {'period', 'frequency'}
        Abscissa of the dispersion panel.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, (ax_s, ax_d) = plt.subplots(2, 1, figsize=(8, 8))
    else:
        ax_s, ax_d = ax
        fig = ax_s.figure

    if 'f_band' in result:
        ax_s.plot(result['f_band'], result['s_band'] / np.max(np.abs(result['s_band'])),
                  color='k', lw=0.9, label='Re[cross-spectrum], splined')
    ax_s.axhline(0, color='0.6', lw=0.5)

    kept = np.asarray(result.get('f_cross', []), dtype=float)
    allx = np.asarray(result.get('f_cross_all', kept), dtype=float)
    dropped = np.setdiff1d(allx, kept)
    if dropped.size:
        ax_s.plot(dropped, np.zeros_like(dropped), 'x', ms=6, color='tab:red',
                  mew=1.4, zorder=5,
                  label=f'dropped ({dropped.size}): group vel. inconsistent')
    if kept.size:
        ax_s.plot(kept, np.zeros_like(kept), 'o', ms=4.5, color='tab:green',
                  mec='k', mew=0.5, zorder=6, label=f'used ({kept.size})')

    ax_s.set_xlabel('Frequency (Hz)')
    ax_s.set_ylabel('Re, normalized')
    ax_s.set_title('Bessel zero crossings of the real cross-spectrum'
                   + (f" | J0 corr = {result['bessel_corr']:.2f}"
                      if np.isfinite(result.get('bessel_corr', np.nan)) else ''),
                   fontsize=10)
    ax_s.legend(fontsize=8, loc='upper right')

    def _x(f):
        return 1.0 / np.asarray(f) if xaxis == 'period' else np.asarray(f)

    for cd in result.get('candidates', [])[:max_branches]:
        ax_d.plot(_x(cd['f']), cd['c'], marker='^', ms=3, lw=0.4,
                  color='0.72', zorder=1)
    if result.get('ok'):
        ax_d.plot(_x(result['f']), result['c'], marker='o', ms=5, color='k',
                  lw=1.2, zorder=4,
                  label=f"selected phase vel. (branch {result['branch']})")
    if T_vg is not None and vg is not None:
        x = np.asarray(T_vg) if xaxis == 'period' else 1.0 / np.asarray(T_vg)
        ax_d.plot(x, vg, color='tab:blue', lw=1.4, zorder=3, label='group vel. (FTAN)')

    ax_d.set_xscale('log')
    ax_d.set_xlabel('Period (s)' if xaxis == 'period' else 'Frequency (Hz)')
    ax_d.set_ylabel('Velocity (km/s)')
    if result.get('ok'):
        ax_d.set_ylim(0, 1.6 * np.nanmax(result['c']))
    ax_d.legend(fontsize=8)
    if result.get('ambiguous'):
        ax_d.set_title(f"AMBIGUOUS: {result['n_survivors']} admissible branches",
                       color='tab:red', fontsize=10)
    fig.tight_layout()
    return fig


__all__ = [
    'measure_phase_velocity_aki',
    'measure_phase_velocity_fry',
    'phase_from_group_integration',
    'select_branch',
    'trim_crossing_sequence',
    'trim_edge_outliers',
    'branch_phase_velocity',
    'branch_group_velocity',
    'real_spectrum',
    'smooth_real_spectrum',
    'find_zero_crossings',
    'bessel_fit_quality',
    'quadrature_offset',
    'crossings_of',
    'spectral_resolution_check',
    'group_curve_to_frequency',
    'build_reference_curve',
    'save_dispersion',
    'plot_aki',
    'J0_ZEROS',
]