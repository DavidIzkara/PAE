# zarr_utils.py
"""
Utility functions for reading, writing, and managing physiological data
stored in Zarr format.

Compatible with Zarr v2 (appendable 1D datasets, Blosc compression).
"""

from __future__ import annotations

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
# 🔧 Configuration
#     Set up the compressor and chunking strategy
#     that will be used in our Zarr files.
# ---------------------------------------------------------------------
_DEFAULT_COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
_DEFAULT_CHUNK = (60_000,)  # ~5 min at 200 Hz
STORE_PATH = os.path.join("results", "BOX01_20251024.zarr")
FRAME_SIGNAL = "signals/Intellivue/"
FRAME_SIGNAL_DEMO = "signals/Demo/"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def generate_uid(vital_path:str)-> str:
    #Same .vital → same UID.
    base = os.path.basename(vital_path)  # ex: "n5j8vrrsb_250127_100027.vital"
    digest = hashlib.sha256(base.encode("utf-8")).digest()
    token = base64.b32encode(digest[:6]).decode("utf-8").rstrip("=")
    return "UI" + token[:8]


VISIBLE_ALGORITHMS = {
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
#   Basic functions that will
#   help us manage other functions
# ---------------------------------------------------------------------
def open_root(store_path: str) -> zarr.hierarchy.Group:
    """Open (or create) a Zarr container."""
    os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
    return zarr.open(store_path, mode="a")

# Convert time measured in seconds since the
# epoch 1700-01-01 into a Python datetime
def epoch1700_to_datetime(ts_seconds: float) -> datetime:
    """Convert VitalDB-style seconds since 1700-01-01 UTC to datetime."""
    epoch_1700 = datetime(1700, 1, 1, tzinfo=timezone.utc)
    return epoch_1700 + timedelta(seconds=float(ts_seconds))

def string_to_epoch1700(date_str: str, formato: str = TIMESTAMP_FORMAT) -> Union[float, None]:
    """
    Converts a datetime string (assumed to be in UTC)
    into seconds elapsed since 1700-01-01 UTC.

    Args:
        date_str: The datetime string (e.g., '2025-11-20 13:03:00').
        formato: The expected string format 
                 (default: '%Y-%m-%d %H:%M:%S').

    Returns:
        A float representing the number of seconds elapsed 
        since 1700-01-01, or None if parsing fails or the date 
        precedes the 1700 epoch.
    """

    try:
        dt_naive = datetime.strptime(date_str, formato)
        
        dt_utc = dt_naive.replace(tzinfo=timezone.utc)
        
    except ValueError as e:
        print(f"[ERROR] Incorrect datetime format: {e}")
        return None

    epoch_1700 = datetime(1700, 1, 1, tzinfo=timezone.utc)

    if dt_utc < epoch_1700:
        return None
        
    time_difference: timedelta = dt_utc - epoch_1700

    return time_difference.total_seconds()


def normalize_signal_path(track: str) -> str:
    """
    Normalizes a track string to ensure it has a valid full path.

    Rules:
      - If it starts with 'signals/' or 'pred/' → returned as-is.
      - If it starts with 'Intellivue/' → prepend 'signals/'.
      - If no prefix is present → assume 'signals/Intellivue/<track>'.

    Examples:
        ECG_HR                    → signals/Intellivue/ECG_HR
        Intellivue/ECG_HR         → signals/Intellivue/ECG_HR
        signals/Intellivue/ECG_HR → unchanged
        pred/CO/beatwise          → unchanged
    """

    if track.startswith("signals/") or track.startswith("pred/"):
        return track
    
    # Case 1: starts with "Intellivue/"
    if track.startswith("Intellivue/"):
        return f"signals/{track}"
    
    # Case 2: only short name
    return f"signals/Intellivue/{track}"


# ---------------------------------------------------------------------
# 🧩 STRUCTURE MANAGEMENT
#    # Functions that help manipulate the hierarchy of groups and datasets.
# ---------------------------------------------------------------------
def safe_group(root: zarr.hierarchy.Group, path: str) -> zarr.hierarchy.Group:
    """Creates (if needed) and returns the subgroup inside the root."""
    parts = [p for p in path.split("/") if p]
    g = root
    for p in parts:
        g = g.require_group(p)
    return g
# path ->  /aaa/bbb/ccc
# root -> root group of the .zarr file, the top-level object of the hierarchy
# Takes the root and keeps creating subgroups if they do not exist
# root.require_group(path_part) is equivalent to the Linux 'mkdir':
# if the group exists, it is returned; if it does not, it is created.
# Best suited for write operations

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
# This only returns the subgroup if it already exists; otherwise it does NOT create anything new.
# Better suited for read operations
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
# group -> This is a subgroup within the hierarchy.
# Conceptually, it behaves like a directory inside the Zarr store.
# name -> Name of the 1D array we want to create or retrieve.
# dtype -> Data type used to store the array values.
# fill -> Default value for newly allocated positions.
# compressor -> Compression algorithm applied to the dataset.
# chunks -> Chunk size used for storage.
# Creates or returns a resizable 1D array inside the group.
# shape=(0,) -> The array starts empty.
# maxshape=(None,) -> Allows the array to grow indefinitely.
# overwrite=False -> Will NOT overwrite the dataset if it already exists.
# This is essential for incremental writing of data or for reading
# time-series streams efficiently.


def append_1d(arr: zarr.core.Array, data: np.ndarray) -> None:
    """Append 1D data to a resizable Zarr array."""
    if data.size == 0:
        return
    n_old = arr.shape[0]
    n_new = n_old + len(data)
    arr.resize(n_new)
    arr[n_old:n_new] = data
# arr -> The target Zarr array to which data will be appended.
# data -> A 1D NumPy array containing the samples to append.
#
# Appends 1D data to a resizable Zarr array.
# If there is no data to append, the function simply returns.
#
# arr.shape -> Retrieves the current size of the array.
# The function computes the new size and resizes the array before writing.
#
# Used for incremental writing operations.


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
# parent_group -> The group in which the signal pair should be created or retrieved.
# signal_path -> The full hierarchical path of the signal inside the Zarr structure.
# dtype -> The data type to use for the signal values.
# Returns a tuple with two Zarr arrays:
#   - one for the timestamps in milliseconds
#   - one for the signal values
# Used both for initializing write operations and for reading signals in algorithms.

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

# Extracts a time-window segment of a signal between a start and end time.
# t_abs_ms -> Array of absolute timestamps in milliseconds.
# vals -> Array of signal values.
# start_s -> Start time in seconds.
# end_s -> End time in seconds.
# Returns the sliced arrays of timestamps and values
# corresponding to the specified time window.


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
# Performs a recursive traversal of the entire Zarr hierarchy.
# Returns all arrays found, but skips groups.

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
    Exports the specified tracks from the .vital file into the .zarr container
    using the structure:

        signals/<track>/time_ms
        signals/<track>/value

    APPEND mode: only adds samples where ts_ms > last_ts.
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

    # save the UID (last path folder)
    if "patient_uid" not in root.attrs:
        root.attrs["patient_uid"] = os.path.basename(zarr_path)

    signals_root = safe_group(root, "signals")

    written_tracks = 0
    total_added = 0

     

    for track in tracks:
        # 1) read the track from the .Vital as a DataFrame
        try:
            df = vf.to_pandas(track_names=track, interval=0, return_timestamp=True)
        except Exception as e:
            print(f"[WARN] Couldn't read '{track}' from the .Vital: {e}")
            continue

        if df is None or df.empty or track not in df.columns:
            print(f"[WARN] Track '{track}' empty or without a DataFrame column, skipping.")
            continue

        # 2) Time and values
        ts = df["Time"].to_numpy(dtype=float)
        vals = df[track].to_numpy(dtype=float)

        # Clean NaNs
        mask = np.isfinite(vals)
        ts = ts[mask]
        vals = vals[mask]

        if ts.size == 0:
            print(f"[WARN] Track '{track}' has no valid samples, skipping."
)
            continue

        # 3) Convert to ms
        ts_ms = np.rint(ts * 1000.0).astype("int64")
        vals_f32 = vals.astype("float32")

        # 4) Group in Zarr: signals/<track>/time_ms + value
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


        # 5) APPEND Mode: Only add new samples (ts_ms > last_ts)
        if ds_time.size > 0:
            last_ts = int(ds_time[-1])
            mask_new = ts_ms > last_ts
            ts_ms = ts_ms[mask_new]
            vals_f32 = vals_f32[mask_new]

        if ts_ms.size == 0:
            print(f"[INFO] Track '{track}': no new samples to add.")
            continue

        # 6) Efective Append 
        append_1d(ds_time, ts_ms)
        append_1d(ds_val, vals_f32)

        grp.attrs["track"] = track
        grp.attrs.setdefault("units", "")
        grp.attrs.setdefault("notes", "")

        written_tracks += 1
        total_added += ts_ms.size
        print(f"[OK] {track}: +{ts_ms.size} samples")

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

    # Use relative time only for display purposes, as a local variable
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


def read_signal(
    zarr_path: str,
    track: str,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None
) -> Optional[pd.DataFrame]:
    """
    High-level helper function to read a signal easily.
    """
    root = open_root(zarr_path)
    time_ms, value = load_track(root, track)

    if time_ms is None:
        return None
        # Apply time-windowing (seconds relative to the first timestamp)
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


def read_multiple_signals(
    zarr_path: str,
    tracks: List[str],
    start_s: Optional[float] = None,
    end_s: Optional[float] = None
) -> Dict[str, pd.DataFrame]:
    """
    Reads multiple signals by repeatedly using leer_senyal().

    Args:
        zarr_path: Path to the .zarr container.
        tracks: List of signal paths.
        start_s: Start of the time window (optional, seconds since the first sample).
        end_s: End of the time window (optional, seconds since the first sample).

    Returns:
        A dict where each key is a track_path and each value is a DataFrame
        containing the columns ['time_ms', 'value'].
    """

    resultado = {}
    
    for track in tracks:
        try:
            data = read_signal(zarr_path, track, start_s, end_s)
            if data is not None:
                resultado[track] = data
        except KeyError as e:
            print(f"[WARN] Couldn't read{track}: {e}")
            continue
    
    return resultado

def write_signal(
    zarr_path: str,
    track: str,
    timestamps_ms: np.ndarray,
    values: np.ndarray,
    metadata: Optional[Dict] = None
) -> Optional[bool]:  #Return None if misses
    """
    Writes a signal using get_or_create_signal_pair() and append_1d().
    
    Args:
        zarr_path: Path to the .zarr container.
        track_path: Destination path (e.g., "Intellivue/PLETH").
        timestamps_ms: Timestamps in milliseconds.
        values: Signal values.
        metadata: Optional dictionary containing metadata.
    """

    if timestamps_ms.size != values.size:
        raise ValueError(f"Timestamps and values must have the same length")
    
    root = open_root(zarr_path)
    signals_root = safe_group(root, "signals")
    
    # Use get_or_create_signal_pair to obtain the arrays
    time_arr, data_arr = get_or_create_signal_pair(signals_root, track)
    
    # Use append_1d to add data
    append_1d(time_arr, timestamps_ms.astype(np.int64))
    append_1d(data_arr, values.astype(np.float32))
    
    # save metadata if given
    if metadata:
        path_parts = track.split("/")
        grp = signals_root
        for part in path_parts:
            grp = grp.require_group(part)
        
        for k, v in metadata.items():
            grp.attrs[k] = v
    
    print(f"✅ {timestamps_ms.size} samples written in signals/{track}")
    return True


def write_prediction(
    zarr_path: str,
    pred_name: str, 
    timestamps_ms: np.ndarray,
    values: np.ndarray,
    modelo_info: Optional[Dict],
    timestamps_fin_ms: Optional[np.ndarray]
) -> None:
    """
    Writes predictions into the Zarr store using the structure:
    ROOT/predictions/MODEL_NAME/PRED_NAME/time_ms
    ROOT/predictions/MODEL_NAME/PRED_NAME/value
    """

    multi_timestamp = False

    if timestamps_ms.size != values.size:
        raise ValueError(f"Timestamps and values must have the same size")
    
    if timestamps_fin_ms is not None:
        multi_timestamp = True

    root = open_root(zarr_path)
    
    # 1. Retrieve the algorithm/model name (e.g., "Shock Index")
    # Apply validation and fall back to a default name if missing.

    model_name = modelo_info.get("model") if modelo_info and isinstance(modelo_info, dict) and "model" in modelo_info else "Unknown_Model"
    
    # 2. Build the FULL target path
    # Example: "predictions/Shock Index/SI"

    full_group_path = f"predictions/{model_name}/{pred_name}"
    
    # 3. Create the full group safely
    # We assume that safe_group can create nested paths (predictions, Shock Index, SI)
    # 'grp' is the final Zarr group, which will be named 'SI'.

    grp = safe_group(root, full_group_path)
    
    # 4. Create or open the 'time_ms' and 'value' datasets INSIDE the group 'grp' ('SI')
    # We use get_or_create_1d directly to avoid creating unwanted nested folder structures.
    # (Assumes that get_or_create_1d and append_1d are available in utils_zarr.py)
  
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
    
    # 5. Model metadata: added to the 'SI' group (grp)
    if modelo_info:
        for key, val in modelo_info.items():
            grp.attrs[f"model_{key}"] = val
        grp.attrs["prediction_created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"✅ Prediction '{pred_name}' saved in '{full_group_path}': {timestamps_ms.size} samples")

def read_group_zattrs(zarr_path: str, grupo_path: str) -> Dict[str, Any]:
    """
    Extracts and returns the contents of the .zattrs metadata file
    from a specific group inside the Zarr container.

    Args:
        zarr_path: Path to the Zarr file (e.g., "session_data.zarr")
        group_path: Internal path of the group (e.g., "signals/Intellivue/ECG_HR")

    Returns:
        A dictionary containing the metadata. Returns {} if the group
        does not exist or contains no metadata.
    """

    
    # Open the Zarr container to obtain the root group.
    try:
        root = open_root(zarr_path)
    except Exception as e:
        # In a real environment, we would raise an error or return a failure code.
        print(f"[ERROR] Couldn't open Zarr file '{zarr_path}': {e}")
        return None

    # Navigate to the specific group
    target_group = get_group_if_exists(root, grupo_path)

    if target_group is None:
        print(f"[WARN] Group '{grupo_path}' wasn't found.")
        return None

    # Access and return the metadata (.zattrs)
    return dict(target_group.attrs)
def get_track_names_simplified(zarr_path: str) -> List[str]:
    """
    Retrieves the available track paths and returns only the signal name (e.g., 'ECG_HR')
    by extracting the last component of each full path.

    Args:
        zarr_path: Path to the Zarr file (e.g., "results/session_data.zarr")

    Returns:
        A list of strings containing the simplified track names.
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
def obtain_info_zarr(zarr_path: str) -> Dict:
    """
    Summary of the Zarr contents using list_available_tracks() and load_track().

    Returns:
        A dict containing 'signals', 'predictions', 'n_signals', etc.
    """

  
    signals, preds = list_available_tracks(zarr_path)
    
    root = open_root(zarr_path)
    
    # Compute the approximate duration of the first signal
    total_duration_s = 0
    if signals:
        try:
            t_abs, vals = load_track(root, signals[0])
            if t_abs is not None and t_abs.size > 0:
                total_duration_s = (int(t_abs[-1]) - int(t_abs[0])) / 1000.0
        except Exception:
            pass
    
    return {
        'signals': signals,
        'predictions': preds,
        'n_signals': len(signals),
        'n_predictions': len(preds),
        'total_duration_s': total_duration_s,
        'metadata': dict(root.attrs) if hasattr(root, 'attrs') else {}
    }


def write_multiple_signals(
    zarr_path: str,
    datos_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> None:
    """
    Writes multiple signals by repeatedly calling write_signal().
    Args:
        datos_dict: Dict where each key = track_path and each value = (timestamps_ms, values)
    """

    for track_path, (timestamps_ms, values) in datos_dict.items():
        try:
            write_signal(zarr_path, track_path, timestamps_ms, values)
        except Exception as e:
            print(f"[ERROR] Couldn't write {track_path}: {e}")
            continue
    
    print(f"✅ Batch completed: {len(datos_dict)} processed signals")


def export_time_window(
    zarr_path: str,
    output_path: str,
    track_paths: List[str],
    start_s: float,
    end_s: float
) -> None:
    """Exports a time window of multiple signals into a new Zarr file.

    Args:
        zarr_path: Source Zarr file.
        output_path: Destination Zarr file.
        track_paths: List of signal paths to export.
        start_s: Start of the time window (seconds).
        end_s: End of the time window (seconds).
    """

    # Read window's data
    data = read_multiple_signals(zarr_path, track_paths, start_s, end_s)
    
    # Prepare data for batch writing
    data_batch = {}
    for track, data in data.items():
        # Remove "signals/" from the path if present
        clean_track = track.replace("signals/", "")
        data_batch[clean_track] = (data['t_abs_ms'].values, data['values'].values)
    
    # Write in the new Zarr
    write_multiple_signals(output_path, data_batch)
    
    print(f"✅ Exported window [{start_s}s - {end_s}s] a {output_path}")
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
    Computes the union of all REQUIRED_TRACKS from the selected algorithms
    and calls vital_to_zarr a single time to export/update them.

    `algorithms_catalog` is a dict of the form:
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
            print(f"[WARN] Unknown Algorithm: {name}")
            continue
        all_tracks.update(info["required_tracks"])

    if not all_tracks:
        print("[WARN] There are no tracks to export (no valid algorithm).")
        return

    print("\n[DISPATCH] Selected Algorithms:", algo_names)
    print("[DISPATCH] Tracks to export/update:")
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
    Executes the specified algorithms on the Zarr file and returns a dict
    {algorithm_name: instance_or_None}.
    If 'cardiac_output' is detected and it has the attributes t_last_ms and co_last,
    its prediction is stored.
    """

    results: Dict[str, Any] = {}

    for name in algo_names:
        info = algorithms_catalog.get(name)
        if info is None:
            print(f"[WARN] Unknown Algorithm: {name}")
            continue

        runner = info["runner"]

        try:
            algo_instance = runner(zarr_path)
            results[name] = algo_instance

            # Specific Example for 'cardiac_output'
            if (
                name == "cardiac_output"
                and getattr(algo_instance, "t_last_ms", None) is not None
                and getattr(algo_instance, "co_last", None) is not None
            ):
                t_arr = np.asarray([algo_instance.t_last_ms], dtype=np.int64)
                v_arr = np.asarray([algo_instance.co_last], dtype=np.float32)

                write_prediction(
                    zarr_path=zarr_path,
                    pred_name="cardiac_output",
                    timestamps_ms=t_arr,
                    values=v_arr,
                    modelo_info={"source": "dispatcher", "algo": "cardiac_output"},
                )

                print(
                    f"[ALG-STORE] {name}: last point saved "
                    f"(t_ms={algo_instance.t_last_ms}, value={algo_instance.co_last:.2f}) in 'predictions/cardiac_output'"
                )

        except ValueError as e:
            print(f"[WARN] Couldn't compute'{name}': {e}")
            results[name] = None

    return results
