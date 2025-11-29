# zarr_utils.py
"""
Utility functions for reading, writing, and managing physiological data
stored in Zarr format.

Compatible with Zarr v2 (appendable 1D datasets, Blosc compression).
"""

import hashlib
import os
import time
import numpy as np
import pandas as pd
import zarr
import base64
from numcodecs import Blosc
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

# ---------------------------------------------------------------------
# 🔧 Configuració
#      Preparem el compressor i el chunking 
#      que utilitzarem en els nostres fitxers Zarr.
# ---------------------------------------------------------------------
_DEFAULT_COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
_DEFAULT_CHUNK = (60_000,)  # ~5 min at 200 Hz
STORE_PATH = os.path.join("results", "BOX01_20251024.zarr")
FRAME_SIGNAL = "signals/Intellivue/"
FRAME_SIGNAL_DEMO = "signals/Demo/"
FORMATO_TIMESTAMP = "%Y-%m-%d %H:%M:%S"


def generar_uid(vital_path:str)-> str:
    #El mateix .vital → sempre el mateix UID.
    base = os.path.basename(vital_path)  # ex: "n5j8vrrsb_250127_100027.vital"
    digest = hashlib.sha256(base.encode("utf-8")).digest()
    token = base64.b32encode(digest[:6]).decode("utf-8").rstrip("=")
    return "UI" + token[:8]


ALGORITMOS_VISIBLES = {
    'Shock Index'
    'Driving Pressure'
    'Dynamic Compliance'
    'ROX Index'
    'Temp Comparison'
    'Cardiac Output'
    'Systemic Vascular Resistance'
    'Cardiac Power Output'
    'Effective Arterial Elastance'
}



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

def string_to_epoch1700(date_str: str, formato: str = FORMATO_TIMESTAMP) -> Union[float, None]:
    """
    Convierte una cadena de texto de fecha/hora (asumiendo que es UTC) 
    a segundos transcurridos desde 1700-01-01 UTC.
    
    Args:
        date_str: La cadena de fecha/hora (ej: '2025-11-20 13:03:00').
        formato: El formato de la cadena (por defecto: '%Y-%m-%d %H:%M:%S').
        
    Returns:
        Número de segundos (float) o None si falla o la fecha es anterior a 1700.
    """
    try:
        dt_naive = datetime.strptime(date_str, formato)
        
        dt_utc = dt_naive.replace(tzinfo=timezone.utc)
        
    except ValueError as e:
        print(f"[ERROR] Formato de fecha incorrecto: {e}")
        return None

    epoch_1700 = datetime(1700, 1, 1, tzinfo=timezone.utc)

    if dt_utc < epoch_1700:
        return None
        
    time_difference: timedelta = dt_utc - epoch_1700

    return time_difference.total_seconds()


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
    if track.startswith("signals/") or track.startswith("pred/"):
        return track
    
    # Cas 2: comença per "Intellivue/"
    if track.startswith("Intellivue/"):
        return f"signals/{track}"
    
    # Cas 3: només nom curt
    return f"signals/Intellivue/{track}"


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
# path ->  /algo/algo/algo
# root -> grup arrel del .zarr. l'objecte superior de tota la jerarquia
# Agafa el root i va creant els subgrups si no existeixen
# root.require_group(part del path) es una funcio que equival al
# mkdir de linux. Si existeix el retorna, si no existeix el crea.
# Millor per escriptura
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
# Aquest solament retorna si existeix pero si no NO crea res de nou.
# Millor per lectura
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
# group -> És un subgrup intern dins d'aquesta jerarquia. 
# És simplement un directori dins el Zarr.
# name -> nom del array 1D que volem crear o retornar.
# dtype -> Tipus de dades que volem emmagatzemar.
# fill -> Valor per defecte per a les noves posicions.
# compressor -> Tipus de compressió que volem utilitzar.
# chunks -> Mida dels chunks per a l'emmagatzematge.
# Crear o retornar un array 1D redimensionable dins el grup
# shape=(0,) -> Comença l'array buit
# maxshape=(None,) -> Permet redimensionar l'array infinitament.
# overwrite=False -> No sobreescriu si ja existeix.
#Clau per començar una escriptura incremental de dades o per llegir dades.

def append_1d(arr: zarr.core.Array, data: np.ndarray) -> None:
    """Append 1D data to a resizable Zarr array."""
    if data.size == 0:
        return
    n_old = arr.shape[0]
    n_new = n_old + len(data)
    arr.resize(n_new)
    arr[n_old:n_new] = data
# arr ->
# data ->
# Afegeix dades 1D a un array Zarr redimensionable.
# Si no hi ha dades per afegir, simplement retorna.
# arr.shape -> Obtén la mida actual de l'array.
# Recalcula la abans d'afegir les noves dades en l'array.
# Per escritura

def get_or_create_signal_pair(
    parent_group: zarr.hierarchy.Group,
    signal_path: str,
    dtype="f4",
) -> tuple[zarr.core.Array, zarr.core.Array]:
    """
    Create or retrieve the pair (time_ms, values) for a signal.
    Example: "Intellivue/PLETH" → "Intellivue/PLETH_time_ms", "Intellivue/PLETH"
    """
    parts = signal_path.split("/")
    *grp_parts, var_name = parts
    g = parent_group
    for p in grp_parts:
        g = g.require_group(p)

    time_arr = get_or_create_1d(g, f"{var_name}_time_ms", dtype="i8", fill=-1)
    data_arr = get_or_create_1d(g, f"{var_name}", dtype=dtype, fill=np.nan)
    return time_arr, data_arr
# parent_group -> El grup on volem crear o obtenir el par de senyals.
# signal_path -> El camí complet del senyal dins la jerarquia.
# dtype -> El tipus de dades per als valors del senyal.
# Retorna un tuple amb dos arrays Zarr: un per als temps en ms i un altre per als valors del senyal.
# Tant per iniciar una escritura o per llegir per algoritmes.

# ---------------------------------------------------------------------
# 🩺 READING / NAVIGATION
# ---------------------------------------------------------------------
def load_track(root: zarr.hierarchy.Group, signal: str):
    """Return (t_abs_ms, values) for a signal."""
    track_path = normalize_signal_path(signal)
    t_key = f"{track_path}/time_ms"
    v_key = f"{track_path}/value"
    
    if t_key not in root or v_key not in root:
        return None, None

    t_abs_ms = root[t_key][:].astype(np.int64)
    vals = root[v_key][:].astype(np.float32)
    return t_abs_ms, vals

def slice_by_seconds(t_abs_ms, vals, start_s, end_s):
    """Return a time-windowed segment of signal data using seconds relative to the first sample."""
    if t_abs_ms is None or t_abs_ms.size == 0:
        return t_abs_ms, vals

    t0 = int(t_abs_ms[0])
    rel_ms = t_abs_ms - t0

    start_ms = int(start_s * 1000)
    end_ms = int(end_s * 1000)

    i0 = np.searchsorted(rel_ms, start_ms, side="left")
    i1 = np.searchsorted(rel_ms, end_ms, side="left")
    return t_abs_ms[i0:i1], vals[i0:i1]

#Extrau un segment temporal d'un senyal entre un inici i un final.
# t_abs_ms -> Array de temps absoluts en mil·lisegons.
# vals -> Array de valors del senyal.
# start_s -> Temps d'inici en segons.
# end_s -> Temps final en segons.
# Retarna trams talla dels arrays de temps relatius
# i valors corresponents dins de la finestra especificada.

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
#És un recorregut recursiu per tota la jerarquia Zarr
#Retorna tots els arrays, però no els grups

def list_available_tracks(zarr_path):
    """Return lists of available signal and prediction tracks."""
    root = open_root(zarr_path)

    signals = []
    preds = []
    
    if "signals" in root:
        arrs = walk_arrays(root["signals"], base="signals")
        signals = [
            p.replace("/value", "")  
            for p in arrs
            if p.endswith("/value")
        ]

    if "pred" in root:
        arrs = walk_arrays(root["pred"], base="pred")
        preds = [
            p.replace("/value", "")  
            for p in arrs
            if p.endswith("/value")
        ]
    return signals, preds

def track_exists(signals, track_name: str) -> bool:
    if "/" not in track_name:
        return False
    vendor, track = track_name.split("/", 1)
    return vendor in signals and track in signals[vendor]


# ---------------------------------------------------------------------
# 🧬 VITAL → ZARR CONVERSION
# ---------------------------------------------------------------------
from vitaldb import VitalFile

def vital_to_zarr(
    vital_file: str,
    zarr_path: str,
    tracks: list[str],
    window_secs: float | None = None,
    chunk_len: int = 30000,
) -> None:
    """
    Exporta les tracks indicades del .vital al .zarr en format:

        signals/<track>/time_ms
        signals/<track>/value

    Mode APPEND: només afegeix mostres amb ts_ms > last_ts.
    """
    if not os.path.exists(vital_file):
        raise FileNotFoundError(f"Missing .vital: {vital_file}")

    vf = VitalFile(vital_file)

    root = open_root(zarr_path)

    # metadata
    root.attrs.setdefault("schema", "v1")
    root.attrs.setdefault("created_by", "zarr_utils")
    root.attrs.setdefault("time_origin", "epoch1700_ms")
    root.attrs["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Guardem el UID (última carpeta del path)
    if "patient_uid" not in root.attrs:
        root.attrs["patient_uid"] = os.path.basename(zarr_path)

    signals_root = safe_group(root, "signals")

    written_tracks = 0
    total_added = 0

     

    for track in tracks:
        # 1) Llegim la track del Vital com a DataFrame
        try:
            df = vf.to_pandas(track_names=track, interval=0, return_timestamp=True)
        except Exception as e:
            print(f"[WARN] No s'ha pogut llegir '{track}' del Vital: {e}")
            continue

        if df is None or df.empty or track not in df.columns:
            print(f"[WARN] Track '{track}' buida o sense columna al DataFrame, s'omet.")
            continue

        # 2) Temps i valors
        ts = df["Time"].to_numpy(dtype=float)
        vals = df[track].to_numpy(dtype=float)

        # Netejar NaNs
        mask = np.isfinite(vals)
        ts = ts[mask]
        vals = vals[mask]

        if ts.size == 0:
            print(f"[WARN] Track '{track}' no té mostres vàlides, s'omet.")
            continue

        # 3) Convertim a ms
        ts_ms = np.rint(ts * 1000.0).astype("int64")
        vals_f32 = vals.astype("float32")

        # 4) Grup al Zarr: signals/<track>/time_ms + value
        grp = safe_group(signals_root, track)

        # time_ms
        if "time_ms" in grp:
            ds_time = grp["time_ms"]
        else:
            ds_time = grp.create_dataset(
                "time_ms",
                shape=(0,),
                chunks=(chunk_len,),
                dtype="int64",
                compressor=_DEFAULT_COMPRESSOR,
            )

        # value
        if "value" in grp:
            ds_val = grp["value"]
        else:
            ds_val = grp.create_dataset(
                "value",
                shape=(0,),
                chunks=(chunk_len,),
                dtype="float32",
                compressor=_DEFAULT_COMPRESSOR,
            )


        # 5) Mode APPEND: només afegim mostres noves (ts_ms > last_ts)
        if ds_time.size > 0:
            last_ts = int(ds_time[-1])
            mask_new = ts_ms > last_ts
            ts_ms = ts_ms[mask_new]
            vals_f32 = vals_f32[mask_new]

        if ts_ms.size == 0:
            print(f"[INFO] Track '{track}': no hi ha mostres noves a afegir.")
            continue

        # 6) Append efectiu
        append_1d(ds_time, ts_ms)
        append_1d(ds_val, vals_f32)

        grp.attrs["track"] = track
        grp.attrs.setdefault("units", "")
        grp.attrs.setdefault("notes", "")

        written_tracks += 1
        total_added += ts_ms.size
        print(f"[OK] {track}: +{ts_ms.size} mostres")

    print(f"✅ Updated {zarr_path}: {written_tracks} tracks, {total_added} samples added.")

# ---------------------------------------------------------------------
# 🧠 QUICK SUMMARY
# ---------------------------------------------------------------------
def dump_track(root, track_path, head=5, tail=5):
    """Print quick statistics for a given track."""
    try:
        t_abs, vals = load_track(root, track_path)
    except KeyError:
        print(f"[WARN] Missing {track_path}")
        return

    if t_abs is None or vals is None:
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

    # Fem servir temps relatiu només per mostrar, com a variable local
    if t_abs.size > 0:
        offset_ms = t_abs - int(t_abs[0])
    else:
        offset_ms = t_abs

    print("Head:")
    for ms, v in zip(offset_ms[:head], vals[:head]):
        print(f"  t={int(ms)} ms, v={v}")
    print("Tail:")
    for ms, v in zip(offset_ms[-tail:], vals[-tail:]):
        print(f"  t={int(ms)} ms, v={v}")


def leer_senyal(
    zarr_path: str,
    track: str,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None
) -> Optional[pd.DataFrame]:
    """
    Función de alto nivel para leer un señal fácilmente.
    
    """
    root = open_root(zarr_path)
    time_ms, value = load_track(root, track)

    if time_ms is None:
        return None
    
    # Aplicar ventana temporal (segons des del primer punt)
    if start_s is not None or end_s is not None:
        if start_s is None:
            start_s = 0.0
        if end_s is None:
            end_s = (time_ms[-1] - time_ms[0]) / 1000.0 if time_ms.size > 0 else 0.0
        
        time_ms, value = slice_by_seconds(time_ms, value, start_s, end_s)
    
    df = pd.DataFrame({
        "time_ms": time_ms,
        "value": value,
    })

    return df


def leer_multiples_senyales(
    zarr_path: str,
    tracks: List[str],
    start_s: Optional[float] = None,
    end_s: Optional[float] = None
) -> Dict[str, pd.DataFrame]:
    """Lee múltiples señales usando leer_senyal() repetidamente.
    
     Args:
        zarr_path: Ruta al .zarr
        tracks: Lista de paths de señales
        start_s: Inicio ventana temporal (opcional, segundos desde el primer punto)
        end_s: Fin ventana temporal (opcional, segundos desde el primer punto)
    
    Returns:
        dict donde key=track_path, value=DataFrame con columnas ['time_ms', 'value']
    """
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

def escribir_senyal(
    zarr_path: str,
    track: str,
    timestamps_ms: np.ndarray,
    values: np.ndarray,
    metadata: Optional[Dict] = None
) -> Optional[bool]:  # ✅ CORRECCIÓ: Retornar None si falla
    """Escribe un señal usando get_or_create_signal_pair() y append_1d().
    
    Args:
        zarr_path: Ruta al .zarr
        track_path: Path donde guardar (ej: "Intellivue/PLETH")
        timestamps_ms: Timestamps en milisegundos
        values: Valores del señal
        metadata: Dict opcional con metadatos
    """
    if timestamps_ms.size != values.size:
        raise ValueError(f"Timestamps y values deben tener el mismo tamaño")
    
    root = open_root(zarr_path)
    signals_root = safe_group(root, "signals")
    
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
    modelo_info: Optional[Dict],
    timestamps_fin_ms: np.ndarray = None
) -> None:
    """
    Escribe predicciones en Zarr bajo la estructura:
    ROOT/predictions/MODEL_NAME/PRED_NAME/time_ms
    ROOT/predictions/MODEL_NAME/PRED_NAME/value
    """
    multi_timestamp = False

    if timestamps_ms.size != values.size:
        raise ValueError(f"Timestamps y values deben tener el mismo tamaño")
    
    if timestamps_fin_ms is not None:
        multi_timestamp = True

    root = open_root(zarr_path)
    
    # 1. Obtener el nombre del algoritmo/modelo (Ejemplo: 'Shock Index')
    # Se añade validación y un nombre por defecto.
    model_name = modelo_info.get("model") if modelo_info and isinstance(modelo_info, dict) and "model" in modelo_info else "Unknown_Algorithm"
    
    # 2. Construir la ruta COMPLETA deseada
    # Ejemplo: "predictions/Shock Index/SI"
    full_group_path = f"predictions/{model_name}/{pred_name}"
    
    # 3. Crear el grupo completo de forma segura
    # Asumimos que safe_group puede crear rutas anidadas (predictions, Shock Index, SI)
    # 'grp' es el grupo Zarr final, que se llamará 'SI'.
    grp = safe_group(root, full_group_path)
    
    # 4. Crear o abrir los datasets 'time_ms' y 'value' DENTRO del grupo 'grp' ('SI')
    # Usamos get_or_create_1d directamente para evitar estructuras de carpetas anidadas no deseadas.
    # (Asumo que get_or_create_1d y append_1d están disponibles en utils_zarr.py)
    if multi_timestamp:
        time_ini_arr = get_or_create_1d(grp, "time_ini_ms", dtype="i8", fill=-1)
        time_fin_arr = get_or_create_1d(grp, "time_fin_ms", dtype="i8", fill=-1)
        data_arr = get_or_create_1d(grp, "value", dtype=values.dtype, fill=np.nan)
        append_1d(time_ini_arr, timestamps_ms.astype(np.int64))
        append_1d(time_fin_arr, timestamps_fin_ms.astype(np.int64))
        append_1d(data_arr, values.astype(np.float32))
    else:
        time_arr = get_or_create_1d(grp, "time_ms", dtype="i8", fill=-1)
        data_arr = get_or_create_1d(grp, "value", dtype=values.dtype, fill=np.nan)
        append_1d(time_arr, timestamps_ms.astype(np.int64))
        append_1d(data_arr, values.astype(np.float32))
    
    # 5. Adjuntar los datos
    
    
    # 6. Metadata del modelo: se añade al grupo 'SI' (grp)
    if modelo_info:
        for key, val in modelo_info.items():
            grp.attrs[f"model_{key}"] = val
        grp.attrs["prediction_created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"✅ Predicción '{pred_name}' guardada en '{full_group_path}': {timestamps_ms.size} samples")

def leer_zattrs_de_grupo(zarr_path: str, grupo_path: str) -> Dict[str, Any]:
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
        return None

    # Navegar hasta el grupo específico
    target_group = get_group_if_exists(root, grupo_path)

    if target_group is None:
        print(f"[WARN] El grupo '{grupo_path}' no fue encontrado.")
        return None

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
    
    signal_paths, preds_paths = list_available_tracks(zarr_path)

    all_tracks_paths = signal_paths + preds_paths

    final_names = []
    for track_path in all_tracks_paths:
        parts = track_path.split('/')
        if len(parts) >= 2:
            track_name = parts[-1]
            final_names.append(track_name)
    return final_names
def obtener_info_zarr(zarr_path: str) -> Dict:
    """Resumen del contenido usando list_available_tracks() y load_track().
    Returns:
        dict con 'senyales', 'predicciones', 'n_senyales', etc.
    """
  
    signals, preds = list_available_tracks(zarr_path)
    
    root = open_root(zarr_path)
    
    # Calcular duración aproximada del primer señal
    duracion_total_s = 0
    if signals:
        try:
            t_abs, vals = load_track(root, signals[0])
            if t_abs is not None and t_abs.size > 0:
                duracion_total_s = (int(t_abs[-1]) - int(t_abs[0])) / 1000.0
        except Exception:
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
    """Escribe múltiples señales usando escribir_senyal() repetidamente.
    Args:
        datos_dict: Dict donde key=track_path, value=(timestamps_ms, values)
    """
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
    """Exporta una ventana temporal de múltiples señales a un nuevo Zarr.
    
    
     Args:
        zarr_path: Zarr de origen
        output_path: Zarr de destino
        track_paths: Señales a exportar
        start_s: Inicio ventana
        end_s: Fin ventana
        
        """
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
# ---------------------------------------------------------------------
# DISPATCH HELPERS
# ---------------------------------------------------------------------
def prepare_zarr_for_algorithms(
    vital_path: str,
    zarr_path: str,
    algo_names: List[str],
    algorithms_catalog: Dict[str, Dict[str, Any]],
    window_secs: float | None = None,
) -> None:
    """
    Calcula la unió de totes les REQUIRED_TRACKS dels algoritmes seleccionats
    i crida vital_to_zarr UNA sola vegada per exportar-les/actualitzar-les.

    `algorithms_catalog` és un dict del tipus:
        {
            "cardiac_output": {
                "required_tracks": [...],
                "runner": CardiacOutput,
            },
            ...
        }
    """
    all_tracks: set[str] = set()

    for name in algo_names:
        info = algorithms_catalog.get(name)
        if info is None:
            print(f"[WARN] Algoritme desconegut: {name}")
            continue
        all_tracks.update(info["required_tracks"])

    if not all_tracks:
        print("[WARN] No hi ha tracks a exportar (cap algoritme vàlid).")
        return

    print("\n[DISPATCH] Algoritmes seleccionats:", algo_names)
    print("[DISPATCH] Tracks a exportar/actualitzar:")
    for t in sorted(all_tracks):
        print("   -", t)

    vital_to_zarr(
        vital_file=vital_path,
        zarr_path=zarr_path,
        tracks=sorted(all_tracks),
        window_secs=window_secs,
    )


def run_algorithms_on_zarr(
    zarr_path: str,
    algo_names: List[str],
    algorithms_catalog: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Executa els algoritmes indicats sobre el Zarr i retorna un dict
    {nom_algoritme: instància_o_None}.
    Si detecta 'cardiac_output' amb atributs t_last_ms i co_last, en guarda la predicció.
    """
    results: Dict[str, Any] = {}

    for name in algo_names:
        info = algorithms_catalog.get(name)
        if info is None:
            print(f"[WARN] Algoritme desconegut: {name}")
            continue

        runner = info["runner"]

        try:
            algo_instance = runner(zarr_path)
            results[name] = algo_instance

            # Exemple específic per 'cardiac_output'
            if (
                name == "cardiac_output"
                and getattr(algo_instance, "t_last_ms", None) is not None
                and getattr(algo_instance, "co_last", None) is not None
            ):
                t_arr = np.asarray([algo_instance.t_last_ms], dtype=np.int64)
                v_arr = np.asarray([algo_instance.co_last], dtype=np.float32)

                escribir_prediccion(
                    zarr_path=zarr_path,
                    pred_name="cardiac_output",
                    timestamps_ms=t_arr,
                    values=v_arr,
                    modelo_info={"source": "dispatcher", "algo": "cardiac_output"},
                )

                print(
                    f"[ALG-STORE] {name}: guardat últim punt "
                    f"(t_ms={algo_instance.t_last_ms}, value={algo_instance.co_last:.2f}) a 'predictions/cardiac_output'"
                )

        except ValueError as e:
            print(f"[WARN] No s'ha pogut calcular '{name}': {e}")
            results[name] = None

    return results
