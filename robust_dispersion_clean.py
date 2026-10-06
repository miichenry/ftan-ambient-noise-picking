"""
Eliminacion automatica de saltos (mode-jumps / artefactos) en curvas de
dispersion, siguiendo la tendencia general.

Por que gaussian_filter1d / savgol no sirven
--------------------------------------------------
Son promedios ponderados: si un tramo de la curva salto a un modo o
artefacto equivocado, un filtro lineal lo difumina hacia dentro de la curva
en vez de eliminarlo -- por eso la linea suavizada sigue pasando cerca del
salto en vez de ignorarlo.

Por que un ajuste robusto tipo LOWESS/polinomio tampoco basta
-----------------------------------------------------------------
- LOWESS local con iteraciones de robustez: si el salto es ancho, la
  ventana local centrada dentro del salto no ve suficientes puntos "buenos"
  alrededor -- el ajuste se curva hacia el salto en vez de rechazarlo.
- Polinomio global robusto (IRLS + Tukey): un grado bajo confunde
  curvatura real con un salto; un grado alto tiene suficiente flexibilidad
  para doblarse y seguir el propio salto.

Metodo: deteccion de discontinuidad + comparacion local con vecinos
-----------------------------------------------------------------
1. Se calcula la pendiente local (d Vg / d log10(T)) entre puntos
   consecutivos. Una pendiente MUY por encima de lo normal (umbral robusto
   basado en MAD) marca un "punto de ruptura".
2. Los breakpoints dividen la curva en segmentos contiguos.
3. Para cada segmento se compara su longitud (num. de puntos) con la de
   sus vecinos INMEDIATOS (no con el total de la curva):
   - Segmento interior: candidato a salto si es mas corto que AMBOS
     vecinos (izquierdo y derecho).
   - Segmento de borde (primero o ultimo): candidato si es mas corto que
     su unico vecino.
   Esto evita el fallo de usar una fraccion global fija del total de
   puntos: cuando la tendencia buena esta partida en dos tramos grandes
   (uno antes y otro despues del artefacto), ninguno de los dos por
   separado tiene por que superar el 50% de la curva, pero cada uno SI es
   mas largo que el artefacto que tiene al lado -- que es la comparacion
   que realmente importa. Tambien cubre el caso de un salto de nivel
   PERMANENTE justo al principio o al final de la curva (el segmento del
   borde es mas corto que su unico vecino), que una regla "solo interior"
   o "solo el mas largo de todos" no detectaba.
4. Un segmento candidato se elimina solo si, ademas, su nivel (Vg mediano)
   se desvia del nivel esperado -- interpolado entre sus dos vecinos si es
   interior, o comparado directamente contra su unico vecino si es de
   borde -- por mas de `level_k` veces el ruido tipico punto-a-punto de la
   curva. Esto es lo que distingue un salto real (nivel claramente
   desplazado) de curvatura genuina pronunciada (que vuelve cerca del
   nivel esperado al salir del tramo).
"""

import numpy as np


def remove_dispersion_jumps(T, vg, k_sigma=15.0, level_k=10.0):
    """Elimina segmentos de una curva de dispersion que son saltos a un
    modo/artefacto distinto (en el borde o en medio de la curva), en vez de
    curvatura real de la tendencia.

    Parameters
    ----------
    T, vg : array_like
        Periodo y velocidad de grupo de la curva picked.
    k_sigma : float, optional
        Umbral (en unidades de MAD robusta) de la pendiente local
        ``d(vg)/d(log10 T)`` para marcar un punto como "ruptura". Mas alto
        -> menos sensible; mas bajo -> mas agresivo (mas riesgo de cortar
        curvatura real pronunciada).
    level_k : float, optional
        Cuantas "unidades de ruido tipico" debe desviarse el nivel de un
        segmento candidato respecto al nivel esperado de sus vecinos para
        confirmarlo como salto.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(T_clean, vg_clean)``, en orden ascendente de periodo, sin los
        segmentos identificados como salto.
    """
    T = np.asarray(T, dtype=float)
    vg = np.asarray(vg, dtype=float)
    order = np.argsort(T)
    T, vg = T[order], vg[order]
    n = len(T)
    if n < 5:
        return T, vg

    logT = np.log10(T)

    # ── 1. Puntos de ruptura: pendiente local anomala (MAD robusta) ────────────
    dT = np.diff(logT)
    dT[dT == 0] = np.finfo(float).eps  # evita division por cero si hay periodos duplicados
    d = np.diff(vg) / dT
    mad = np.median(np.abs(d - np.median(d)))
    slope_scale = 1.4826 * mad if mad > 0 else (np.std(d) or 1.0)
    breakpoints = np.where(np.abs(d) > k_sigma * slope_scale)[0]

    bounds = [0] + list(breakpoints + 1) + [n]
    segments = [(bounds[j], bounds[j + 1]) for j in range(len(bounds) - 1)]

    if len(segments) == 1:
        return T, vg  # ninguna ruptura -> nada que eliminar

    # ── 2. Escala de ruido tipico punto-a-punto (para la prueba de nivel) ──────
    vg_diff_mad = np.median(np.abs(np.diff(vg)))
    level_scale = 1.4826 * vg_diff_mad if vg_diff_mad > 0 else (np.std(vg) or 1.0)

    seg_len = [hi - lo for lo, hi in segments]

    keep = np.ones(n, dtype=bool)
    for si, (lo, hi) in enumerate(segments):
        has_left = si > 0
        has_right = si < len(segments) - 1

        # ── 3. Candidato solo si es mas corto que TODOS sus vecinos directos ──
        if has_left and has_right:
            if not (seg_len[si] < seg_len[si - 1] and seg_len[si] < seg_len[si + 1]):
                continue
        elif has_left:
            if not (seg_len[si] < seg_len[si - 1]):
                continue
        elif has_right:
            if not (seg_len[si] < seg_len[si + 1]):
                continue
        else:
            continue  # unico segmento (no deberia pasar, ya cubierto arriba)

        # ── 4. Prueba de nivel contra el/los vecino(s) ─────────────────────────
        seg_level = np.median(vg[lo:hi])
        if has_left and has_right:
            t_before, v_before = logT[lo - 1], vg[lo - 1]
            t_after, v_after = logT[hi], vg[hi]
            seg_mid_t = np.mean(logT[lo:hi])
            expected_level = np.interp(seg_mid_t, [t_before, t_after], [v_before, v_after])
        elif has_left:
            expected_level = vg[lo - 1]
        else:
            expected_level = vg[hi]

        if abs(seg_level - expected_level) > level_k * level_scale:
            keep[lo:hi] = False

    return T[keep], vg[keep]