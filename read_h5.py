"""
Lector de archivos ASDF de cross-correlaciones (NoisePy / SeisNoise style).

Estructura esperada:
    /AuxiliaryData/Allstack_pws/<COMP>   dataset 1D (la forma de onda)
        con atributos: dt, maxlag, dist, azi, baz, latS, lonS, latR, lonR,
        ngood, substack, time, cc_method, comp, [station_source, station_receiver]
"""

import h5py
import numpy as np


def read_ccf_h5(filepath, stack='Allstack_pws', components=None):
    """
    Lee un archivo .h5 (ASDF) de cross-correlaciones de ruido sísmico.

    Parameters
    ----------
    filepath : str
        Ruta al archivo .h5
    stack : str
        Nombre del grupo de stacking dentro de AuxiliaryData
        (por defecto 'Allstack_pws').
    components : list[str] o None
        Lista de componentes a leer (p.ej. ['ZZ', 'RR']).
        Si es None, lee todas las disponibles.

    Returns
    -------
    dict
        {
          'data': {comp: np.ndarray, ...},
          'meta': {comp: {attr: value, ...}, ...},
          'lag_time': np.ndarray (eje temporal, común a todas las componentes)
        }
    """
    result = {'data': {}, 'meta': {}, 'lag_time': None}

    with h5py.File(filepath, 'r') as f:
        group_path = f'AuxiliaryData/{stack}'
        if group_path not in f:
            available = list(f['AuxiliaryData'].keys()) if 'AuxiliaryData' in f else []
            raise KeyError(f"'{group_path}' no existe. Grupos disponibles: {available}")

        grp = f[group_path]
        comp_list = components if components is not None else list(grp.keys())

        for comp in comp_list:
            if comp not in grp:
                print(f"Aviso: componente '{comp}' no encontrada, se omite.")
                continue

            dset = grp[comp]
            data = dset[:]
            attrs = dict(dset.attrs)

            # decodificar bytes -> str si hace falta
            for k, v in attrs.items():
                if isinstance(v, bytes):
                    attrs[k] = v.decode('utf-8')

            result['data'][comp] = data
            result['meta'][comp] = attrs

            # construir el eje de lag time (una sola vez, usando dt/maxlag)
            if result['lag_time'] is None and 'dt' in attrs and 'maxlag' in attrs:
                dt = attrs['dt']
                maxlag = attrs['maxlag']
                npts = len(data)
                result['lag_time'] = np.linspace(-maxlag, maxlag, npts)

    return result


def list_structure(filepath):
    """Imprime la estructura completa del archivo (grupos, datasets y atributos)."""
    with h5py.File(filepath, 'r') as f:
        print("Atributos raíz:")
        for k, v in f.attrs.items():
            print(f"  {k} = {v}")

        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"DATASET {name}  shape={obj.shape}  dtype={obj.dtype}")
            else:
                print(f"GROUP   {name}")

        f.visititems(_visit)


if __name__ == '__main__':
    fpath = 'RI.06436_RI.06490.h5'

    # ver toda la estructura del archivo
    list_structure(fpath)

    # leer solo la componente vertical-vertical (ZZ) y radial-radial (RR)
    result = read_ccf_h5(fpath, components=['ZZ', 'RR'])

    for comp, data in result['data'].items():
        meta = result['meta'][comp]
        print(f"\nComponente {comp}: {data.shape} puntos")
        print(f"  dist = {meta['dist']} km, dt = {meta['dt']} s")
        print(f"  primeros valores: {data[:5]}")

    # ejemplo de gráfico (opcional)
    try:
        import matplotlib.pyplot as plt
        lag = result['lag_time']
        plt.figure(figsize=(8, 4))
        for comp, data in result['data'].items():
            plt.plot(lag, data, label=comp)
        plt.xlabel('Lag time (s)')
        plt.ylabel('Amplitud')
        plt.legend()
        plt.title('Cross-correlaciones apiladas')
        plt.tight_layout()
        plt.savefig('ccf_plot.png', dpi=150)
        print("\nGráfico guardado en ccf_plot.png")
    except ImportError:
        pass