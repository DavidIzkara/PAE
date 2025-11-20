# zarr_utils.py
"""
Utility functions for reading, writing, and managing physiological data
stored in Zarr format.

Compatible with Zarr v2 (appendable 1D datasets, Blosc compression).
"""

import os
import time
import numpy as np
import zarr
import pandas as pd
from numcodecs import Blosc
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple

# ---------------------------------------------------------------------
# 🔧 Configuració
#      Preparem el compressor i el chunking 
#      que utilitzarem en els nostres fitxers Zarr.
# ---------------------------------------------------------------------
_DEFAULT_COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
_DEFAULT_CHUNK = (60_000,)  # ~5 min at 200 Hz
STORE_PATH = os.path.join("results", "BOX01_20251024.zarr")

# ---------------------------------------------------------------------
# 🧱 BASIC HELPERS
#    Funcions bàsiques que ens
#    ajudaran per gestionar les altres funcions.
# ---------------------------------------------------------------------
def open_root(store_path: str) -> zarr.hierarchy.Group:
    """Open (or create) a Zarr container."""
    os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
    return zarr.open(store_path, mode="a")

# Convertim el temps de segons des de 
# l'època 1700-01-01 a datetime
def epoch1700_to_datetime(ts_seconds: float) -> datetime:
    """Convert VitalDB-style seconds since 1700-01-01 UTC to datetime."""
    epoch_1700 = datetime(1700, 1, 1, tzinfo=timezone.utc)
    return epoch_1700 + timedelta(seconds=float(ts_seconds))

def normalize_signal_path(track: str) -> str:
    """
    Normalitza un track perquè tingui un path complet vàlid.
    Regles:
      - Si comença per 'signals/' o 'pred/' → es deixa tal qual.
      - Si comença per 'Intellivue/' → afegeix 'signals/' davant.
      - Si no té prefix → assumeix 'signals/Intellivue/<track>'.
    
    Exemples:
        ECG_HR               → signals/Intellivue/ECG_HR
        Intellivue/ECG_HR    → signals/Intellivue/ECG_HR
        signals/Intellivue/ECG_HR → idem
        pred/CO/beatwise     → idem (no es toca)
    """
    if track.startswith("signals/") or track.startswith("algorithms/"):
        return track
    
    # Cas 2: comença per "Intellivue/"
    if track.startswith("Intellivue/"):
        return f"signals/{track}"
    
    # Cas 3: només nom curt
    return f"signals/Intellivue/{track}"

def normalize_pred_path(pred_name: str) -> str:
    """
    Normalitza el path d'una predicció.
    Regles:
      - Si ja comença per 'pred/' o 'algorithms/' → es deixa igual.
      - Si només és un nom (p. ex. 'CO') → es converteix a 'algorithms/CO'.
      - No obliga vendor ni subcarpetes.
    """
    if pred_name.startswith("algorithms/"):
        return pred_name
    return f"algorithms/{pred_name}"


# ---------------------------------------------------------------------
# 🧩 STRUCTURE MANAGEMENT
#    Funcions que ajuda a manipular la jerarquia de grups i datasets.
# ---------------------------------------------------------------------
def safe_group(root: zarr.hierarchy.Group, path: str) -> zarr.hierarchy.Group:
    """Crea (si cal) i retorna el subgrup dins root."""

    parts = [p for p in path.split("/") if p]
    g = root
    for p in parts:
        g = g.require_group(p)
    return g

def get_group_if_exists(root: zarr.hierarchy.Group, path: str):
    """Return a subgroup if it exists, otherwise None."""
    parts = [p for p in path.split("/") if p]
    g = root
    for p in parts:
        if p in g and isinstance(g[p], zarr.hierarchy.Group):
            g = g[p]
        else:
            return None
    return g

# ---------------------------------------------------------------------
# 💾 ARRAY CREATION & APPENDING
# ---------------------------------------------------------------------
def get_or_create_1d(
    group: zarr.hierarchy.Group,
    name: str,
    dtype="f4",
    fill=np.nan,
    compressor=_DEFAULT_COMPRESSOR,
    chunks=_DEFAULT_CHUNK,
):
    """Return or create a resizable 1D array inside the group."""
    if name in group:
        return group[name]
    return group.create_dataset(
        name,
        shape=(0,),
        dtype=dtype,
        chunks=chunks,
        compressor=compressor,
        fill_value=fill,
        overwrite=False,
    )

def append_1d(arr: zarr.core.Array, data: np.ndarray) -> None:
    """Append 1D data to a resizable Zarr array."""
    if data.size == 0:
        return
    n_old = arr.shape[0]
    n_new = n_old + len(data)
    arr.resize(n_new)
    arr[n_old:n_new] = data

def get_or_create_signal_pair(
    parent_group: zarr.hierarchy.Group,
    signal: str,
    dtype="f4",
) -> tuple[zarr.core.Array, zarr.core.Array]:
    """
    Create or retrieve the pair (time_ms, values) for a signal.
    Example: "Intellivue/PLETH" → "Intellivue/PLETH_time_ms", "Intellivue/PLETH"
    """
    full_path = normalize_signal_path(signal)
    clean = full_path.replace("signals/", "")
    parts = clean.split("/")
    *grp_parts, var_name = parts

    g = parent_group
    for p in grp_parts:
        g = g.require_group(p)
    
    sig_group = g.require_group(var_name)

    time_arr = get_or_create_1d(sig_group, "time_ms", dtype="i8", fill=-1)
    data_arr = get_or_create_1d(sig_group, "value", dtype=dtype, fill=np.nan)
    return time_arr, data_arr

# ---------------------------------------------------------------------
# 🩺 READING / NAVIGATION
# ---------------------------------------------------------------------
def load_track(root: zarr.hierarchy.Group, signal: str):
    """Return (t_abs_ms, t_rel_ms, values) for a signal."""
    track_path = normalize_signal_path(signal)
    t_key = f"{track_path}/time_ms"
    v_key = f"{track_path}/value"  # ✅ CORRECCIÓ: era 'vaules'
    
    if t_key not in root or v_key not in root:
        return None, None, None

    t_abs_ms = root[t_key][:].astype(np.int64)
    vals = root[v_key][:].astype(np.float32)
    if t_abs_ms.size == 0:
        return t_abs_ms, t_abs_ms, vals

    t0 = int(t_abs_ms[0])
    t_rel_ms = (t_abs_ms - t0).astype(np.int64)
    return t_abs_ms, t_rel_ms, vals

def slice_by_seconds(t_rel_ms, vals, start_s, end_s):
    """Return a time-windowed segment of signal data."""
    start_ms = int(start_s * 1000)
    end_ms = int(end_s * 1000)
    i0 = np.searchsorted(t_rel_ms, start_ms, side="left")
    i1 = np.searchsorted(t_rel_ms, end_ms, side="left")
    return t_rel_ms[i0:i1], vals[i0:i1]

def walk_arrays(node, base=""):
    """Recursively list all arrays (not groups) in the hierarchy."""
    out = []
    for name, child in node.items():
        path = f"{base}/{name}" if base else name
        if hasattr(child, "shape") and hasattr(child, "dtype"):
            out.append(path)
        else:
            out.extend(walk_arrays(child, base=path))
    return out

def list_available_tracks(zarr_path):
    """Return lists of available signal and prediction tracks."""
    root = open_root(zarr_path)

    signals = []
    preds = []
    
    if "signals" in root:
        arrs = walk_arrays(root["signals"], base="signals")
        signals = [
            p.replace("/value", "")  # ✅ CORRECCIÓ: era '/values'
            for p in arrs
            if p.endswith("/value")
        ]

    if "pred" in root:
        arrs = walk_arrays(root["pred"], base="pred")
        preds = [
            p.replace("/value", "")  # ✅ CORRECCIÓ: era '/values'
            for p in arrs
            if p.endswith("/value")
        ]
    return signals, preds

def _track_exists(signals, track_name: str) -> bool:
    if "/" not in track_name:
        return False
    vendor, track = track_name.split("/", 1)
    return vendor in signals and track in signals[vendor]


# ---------------------------------------------------------------------
# 🧬 VITAL → ZARR CONVERSION
# ---------------------------------------------------------------------
def vital_to_zarr(
    vital_file,
    zarr_path: str,
    tracks: list[str],
    window_secs: float | None = None,
    chunk_len: int = 30000,
):
    """
    Convert a VitalDB file to Zarr.
    Only writes new, non-empty, non-NaN samples (append mode).
    """
    from vitaldb import VitalFile

    if not os.path.exists(vital_file):
        raise FileNotFoundError(f"Missing .vital: {vital_file}")

    vf = VitalFile(vital_file, tracks)
    root = open_root(zarr_path)

    # metadata
    root.attrs.setdefault("schema", "v1")
    root.attrs.setdefault("created_by", "zarr_utils")
    root.attrs.setdefault("time_origin", "epoch1700_ms")
    root.attrs["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

    signals_root = safe_group(root, "signals")
    total_added = 0
    written_tracks = 0

    for track in tracks:
        try:
            data = vf.to_numpy([track], interval=0, return_timestamp=True)
        except Exception as e:
            print(f"[WARN] Cannot read track {track}: {e}")
            continue
        if data is None or data.size == 0:
            continue

        ts = data[:, 0].astype(np.float64)
        vals = data[:, 1].astype(np.float64)

        if ts.size == 0 or np.all(np.isnan(vals)):
            continue

        ts_ms = np.rint(ts * 1000.0).astype(np.int64)
        vals_f32 = vals.astype(np.float32)

        grp = safe_group(signals_root, track)
        ds_time = grp.require_dataset(
            "time_ms", shape=(0,), chunks=(chunk_len,), dtype="int64", compressor=_DEFAULT_COMPRESSOR
        )
        ds_val = grp.require_dataset(
            "value", shape=(0,), chunks=(chunk_len,), dtype="float32", compressor=_DEFAULT_COMPRESSOR
        )

        # Deduplicate: append only newer timestamps
        if ds_time.shape[0] > 0:
            last_ts = ds_time[-1]
            mask_new = ts_ms > last_ts
            ts_ms = ts_ms[mask_new]
            vals_f32 = vals_f32[mask_new]

        append_1d(ds_time, ts_ms)
        append_1d(ds_val, vals_f32)

        grp.attrs["track"] = track
        grp.attrs.setdefault("units", "")
        grp.attrs.setdefault("notes", "")

        print(f"[OK] {track}: +{ts_ms.size} samples")
        total_added += ts_ms.size
        written_tracks += 1

    print(f"✅ Updated {zarr_path}: {written_tracks} tracks, {total_added} samples added.")

# ---------------------------------------------------------------------
# 🧠 QUICK SUMMARY
# ---------------------------------------------------------------------
def dump_track(root, track_path, head=5, tail=5):
    """Print quick statistics for a given track."""
    try:
        t_abs, t_rel, vals = load_track(root, track_path)
    except KeyError:
        print(f"[WARN] Missing {track_path}")
        return

    print(f"\n[TRACK] {track_path}")
    print(f"Samples: {vals.shape[0]}")
    if vals.size == 0:
        return
    finite = np.isfinite(vals)
    if finite.any():
        fv = vals[finite]
        print(f"min={fv.min():.4f}, max={fv.max():.4f}, mean={fv.mean():.4f}")
    print("Head:")
    for ms, v in zip(t_rel[:head], vals[:head]):
        print(f"  t={int(ms)} ms, v={v}")
    print("Tail:")
    for ms, v in zip(t_rel[-tail:], vals[-tail:]):
        print(f"  t={int(ms)} ms, v={v}")


def leer_senyal(
    zarr_path: str,
    track: str,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None
) -> Optional[pd.DataFrame]:  # ✅ CORRECCIÓ: Type hint
    """
    Función de alto nivel para leer un señal fácilmente.
    """
    root = open_root(zarr_path)
    t_abs_ms, t_rel_ms, values = load_track(root, track)

    # ✅ CORRECCIÓ: Comprovar si és None
    if t_abs_ms is None:
        return None
    
    # Aplicar ventana temporal si se especifica
    if start_s is not None or end_s is not None:
        if start_s is None:
            start_s = 0
        if end_s is None:
            end_s = t_rel_ms[-1] / 1000.0 if t_rel_ms.size > 0 else 0
        
        t_rel_ms, values = slice_by_seconds(t_rel_ms, values, start_s, end_s)
        
        # Recalcular t_abs_ms para la ventana
        if t_rel_ms.size > 0:
            start_idx = np.searchsorted(t_abs_ms, t_abs_ms[0] + int(start_s * 1000))
            end_idx = np.searchsorted(t_abs_ms, t_abs_ms[0] + int(end_s * 1000))
            t_abs_ms = t_abs_ms[start_idx:end_idx]
    
    df = pd.DataFrame({
        't_abs_ms': t_abs_ms,
        't_rel_ms': t_rel_ms,
        'values': values
    })

    return df


def leer_multiples_senyales(
    zarr_path: str,
    tracks: List[str],
    start_s: Optional[float] = None,
    end_s: Optional[float] = None
) -> pd.DataFrame[str, pd.DataFrame]:
    """Lee múltiples señales usando leer_senyal() repetidamente."""
    resultado = {}
    
    for track in tracks:
        try:
            data = leer_senyal(zarr_path, track, start_s, end_s)
            if data is not None:
                resultado[track] = data
        except KeyError as e:
            print(f"[WARN] No se pudo leer {track}: {e}")
            continue
    
    return resultado

def leer_prediccion(
    zarr_path: str,
    pred_name: str,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None
) -> Optional[pd.DataFrame]:
    """
    Lee una predicción almacenada en algorithms/ o pred/, normalizando el path.

    Retorna un DataFrame con:
        t_abs_ms
        t_rel_ms
        values
    o None si no existe.
    """

    root = open_root(zarr_path)
    pred_path = normalize_pred_path(pred_name)

    # Els datasets s'esperen al path:
    #   algorithms/<pred_name>/time_ms
    #   algorithms/<pred_name>/value
    t_key = f"{pred_path}/time_ms"
    v_key = f"{pred_path}/value"

    if t_key not in root or v_key not in root:
        return None

    t_abs_ms = root[t_key][:].astype(np.int64)
    values = root[v_key][:].astype(np.float32)

    if t_abs_ms.size == 0:
        return pd.DataFrame({
            "t_abs_ms": t_abs_ms,
            "t_rel_ms": t_abs_ms,
            "values": values
        })

    # Temps relatiu
    t0 = int(t_abs_ms[0])
    t_rel_ms = (t_abs_ms - t0).astype(np.int64)

    # Finestra temporal opcional
    if start_s is not None or end_s is not None:
        start_s = start_s or 0
        end_s = end_s or (t_rel_ms[-1] / 1000.0)

        t_rel_ms, values = slice_by_seconds(t_rel_ms, values, start_s, end_s)

        # recalcular t_abs_ms per la finestra
        if t_rel_ms.size > 0:
            i0 = np.searchsorted(t_abs_ms, t_abs_ms[0] + int(start_s * 1000))
            i1 = np.searchsorted(t_abs_ms, t_abs_ms[0] + int(end_s * 1000))
            t_abs_ms = t_abs_ms[i0:i1]

    return pd.DataFrame({
        "t_abs_ms": t_abs_ms,
        "t_rel_ms": t_rel_ms,
        "values": values,
    })


def escribir_senyal(
    zarr_path: str,
    track: str,
    timestamps_ms: np.ndarray,
    values: np.ndarray,
    metadata: Optional[Dict] = None
) -> Optional[bool]:  # ✅ CORRECCIÓ: Retornar None si falla
    """Escribe un señal usando get_or_create_signal_pair() y append_1d()."""
    if timestamps_ms.size != values.size:
        return None
    
    root = open_root(zarr_path)
    signals_root = safe_group(root, "signals")
    
    # ✅ CORRECCIÓ: No afegir 'signals/' si ja el té
    # Usar get_or_create_signal_pair para obtener los arrays
    time_arr, data_arr = get_or_create_signal_pair(signals_root, track)
    
    # Usar append_1d para añadir los datos
    append_1d(time_arr, timestamps_ms.astype(np.int64))
    append_1d(data_arr, values.astype(np.float32))
    
    # Guardar metadata si se proporciona
    if metadata:
        # ✅ CORRECCIÓ: Definir path_track correctament
        path_parts = track.split("/")
        grp = signals_root
        for part in path_parts:
            grp = grp.require_group(part)
        
        for k, v in metadata.items():
            grp.attrs[k] = v
    
    print(f"✅ Escritos {timestamps_ms.size} samples en signals/{track}")
    return True


def escribir_prediccion(
    zarr_path: str,
    pred_name: str,
    timestamps_ms: np.ndarray,
    values: np.ndarray,
    modelo_info: Optional[Dict] = None
) -> Optional[bool]:
    """Escribe predicciones en pred/ usando las funciones existentes."""
    if timestamps_ms.size != values.size:
        return None
    
    root = open_root(zarr_path)
    pred_root = safe_group(root, "algorithms")
    
    # Usar get_or_create_signal_pair
    time_arr, data_arr = get_or_create_signal_pair(pred_root, pred_name)
    
    # Usar append_1d
    append_1d(time_arr, timestamps_ms.astype(np.int64))
    append_1d(data_arr, values.astype(np.float32))
    
    # Metadata del modelo
    if modelo_info:
        path_parts = pred_name.split("/")
        grp = pred_root
        for part in path_parts:
            grp = grp.require_group(part)
        
        for key, val in modelo_info.items():
            grp.attrs[f"model_{key}"] = val
    
    print(f"✅ Predicción '{pred_name}' guardada: {timestamps_ms.size} samples")
    return True


def obtener_info_zarr(zarr_path: str) -> Dict:
    """Resumen del contenido usando list_available_tracks() y load_track()."""
    # ✅ CORRECCIÓ: list_available_tracks ja accepta zarr_path
    signals, preds = list_available_tracks(zarr_path)
    
    root = open_root(zarr_path)
    
    # Calcular duración aproximada del primer señal
    duracion_total_s = 0
    if signals:
        try:
            _, t_rel, _ = load_track(root, signals[0])
            if t_rel is not None and t_rel.size > 0:
                duracion_total_s = t_rel[-1] / 1000.0
        except:
            pass
    
    return {
        'senyales': signals,
        'predicciones': preds,
        'n_senyales': len(signals),
        'n_predicciones': len(preds),
        'duracion_total_s': duracion_total_s,
        'metadata': dict(root.attrs) if hasattr(root, 'attrs') else {}
    }


def escribir_batch_senyales(
    zarr_path: str,
    datos_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> None:
    """Escribe múltiples señales usando escribir_senyal() repetidamente."""
    for track_path, (timestamps_ms, values) in datos_dict.items():
        try:
            escribir_senyal(zarr_path, track_path, timestamps_ms, values)
        except Exception as e:
            print(f"[ERROR] No se pudo escribir {track_path}: {e}")
            continue
    
    print(f"✅ Batch completado: {len(datos_dict)} señales procesados")


def exportar_ventana_temporal(
    zarr_path: str,
    output_path: str,
    track_paths: List[str],
    start_s: float,
    end_s: float
) -> None:
    """Exporta una ventana temporal de múltiples señales a un nuevo Zarr."""
    # Leer datos de la ventana
    datos = leer_multiples_senyales(zarr_path, track_paths, start_s, end_s)
    
    # Preparar para escritura batch
    datos_batch = {}
    for track, data in datos.items():
        # Quitar "signals/" del path si existe
        clean_track = track.replace("signals/", "")
        datos_batch[clean_track] = (data['t_abs_ms'].values, data['values'].values)
    
    # Escribir al nuevo zarr
    escribir_batch_senyales(output_path, datos_batch)
    
    print(f"✅ Exportada ventana [{start_s}s - {end_s}s] a {output_path}")

def leer_zattrs_de_grupo(zarr_path: str, grupo_path: str) -> Dict[str, any]:
    """
    Extrae y retorna el contenido del archivo .zattrs (metadatos)
    de un grupo específico dentro del contenedor Zarr.

    Args:
        zarr_path: Ruta al archivo Zarr (ej: "session_data.zarr")
        grupo_path: Path interno del grupo (ej: "signals/Intellivue/ECG_HR")

    Returns:
        Un diccionario (dict) con los metadatos. Retorna {} si el grupo 
        no existe o si no hay metadatos.
    """
    
    # Abrir el contenedor Zarr para obtener el grupo raíz
    try:
        root = open_root(zarr_path)
    except Exception as e:
        # En un entorno real, lanzaríamos un error o devolveríamos un código de fallo.
        print(f"[ERROR] No se pudo abrir el archivo Zarr '{zarr_path}': {e}")
        return {}

    # Navegar hasta el grupo específico
    target_group = get_group_if_exists(root, grupo_path)
    if target_group is None:
        print(f"[WARN] El grupo '{grupo_path}' no fue encontrado.")
        return {}

    # Acceder y devolver los metadatos (.zattrs)
    return dict(target_group.attrs)



def get_track_names_simplified(zarr_path: str) -> List[str]:
    """
    Obtiene los paths de las pistas disponibles y retorna solo el nombre de la señal, (ej: 'ECG_HR')
    extrayendo el penúltimo directorio del path completo.

    Args:
        zarr_path: Ruta al archivo Zarr (ej: "results/session_data.zarr")

    Returns:
        Una lista de strings con los nombres simplificados de las pistas disponibles.
    """
    root = open_root(zarr_path)
    
    signal_paths, preds_paths = list_available_tracks(root)

    all_tracks_paths = signal_paths + preds_paths

    final_names = []
    for track_path in all_tracks_paths:
        parts = track_path.split('/')
        if len(parts) >= 2:
            track_name = parts[-2]
            final_names.append(track_name)
    return final_names
