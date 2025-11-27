import numpy as np
import json
import os
import vitaldb
from collections import defaultdict
from typing import Tuple, List, Dict, Any

# ================= CONFIGURACIÓN =================
# Asegúrate de que este nombre sea correcto
VITAL_FILENAME = 'kuigebjtu_250531_224330.vital' 
HYSTERESIS_VAL = 1.0  # Subimos a 1.0 para filtrar latidos/ruido
OUTPUT_FILENAME = 'semi_markov_model.json'
# =================================================

try:
    from scipy import stats
except Exception:
    stats = None

def get_icp_signal(vf_obj, track_name: str = 'Intellivue/ICP') -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae señal con interval=0 (Resolución Nativa) para coincidir con el predictor.
    """
    # Usamos interval=0 para máxima resolución (no comprimir a 1s)
    try:
        vals = vf_obj.to_numpy(track_name, interval=0)
    except Exception:
        vals = vf_obj.to_numpy('ICP', interval=0)
    
    vals = np.asarray(vals).ravel()
    
    # Intentar sacar timestamps
    timestamps = None
    try:
        res = vf_obj.to_numpy(track_name, interval=0, return_timestamp=True)
        if isinstance(res, tuple) and len(res) >= 2:
            ts_candidate = np.asarray(res[0]).ravel()
            if len(ts_candidate) == len(vals):
                timestamps = ts_candidate
    except Exception:
        pass
        
    if timestamps is None or len(timestamps) != len(vals):
        timestamps = np.arange(len(vals), dtype=float)

    return timestamps, vals

def discretize_icp(values: np.ndarray, thresholds: List[float] = [15.0, 20.0], hysteresis: float = 0.5) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    states = np.zeros_like(values, dtype=int)
    if len(values) == 0: return states

    current_state = 0
    if values[0] >= thresholds[1]: current_state = 2
    elif values[0] >= thresholds[0]: current_state = 1
    states[0] = current_state
    
    t1_low, t1_high = thresholds[0] - hysteresis, thresholds[0] + hysteresis
    t2_low, t2_high = thresholds[1] - hysteresis, thresholds[1] + hysteresis

    for i in range(1, len(values)):
        val = values[i]
        if current_state == 0:
            if val >= t1_high: 
                current_state = 1
                if val >= t2_high: current_state = 2
        elif current_state == 1:
            if val <= t1_low: current_state = 0
            elif val >= t2_high: current_state = 2
        elif current_state == 2:
            if val <= t2_low: 
                current_state = 1
                if val <= t1_low: current_state = 0
        states[i] = current_state
    return states

def estimate_transition_matrix(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid_mask = ~np.isnan(states)
    states = np.asarray(states)[valid_mask]
    n_states = int(np.nanmax(states)) + 1 if len(states) > 0 else 3
    counts = np.zeros((n_states, n_states), dtype=int)
    
    from_states = states[:-1].astype(int)
    to_states = states[1:].astype(int)
    for a, b in zip(from_states, to_states):
        if 0 <= a < n_states and 0 <= b < n_states:
            counts[a, b] += 1

    row_sums = counts.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.divide(counts, row_sums, where=(row_sums != 0))
    P[np.isnan(P)] = 0.0
    return counts, P

def extract_sojourns(states: np.ndarray, timestamps: np.ndarray) -> Dict[int, List[float]]:
    # Vectorized duration calculation (faster/robust)
    change_mask = states[:-1] != states[1:]
    change_indices = np.flatnonzero(change_mask) + 1
    run_starts = np.concatenate(([0], change_indices, [len(states)]))
    
    sojourns = defaultdict(list)
    for i in range(len(run_starts) - 1):
        start, end = run_starts[i], run_starts[i+1]
        state = int(states[start])
        # Calcular duración
        dur = float(timestamps[end-1] - timestamps[start])
        if dur <= 0: dur = 1.0 # Mínimo 1 unidad de tiempo
        sojourns[state].append(dur)
    return dict(sojourns)

def fit_parametric_sojourns(sojourns: Dict[int, List[float]]) -> Dict[int, Dict[str, Any]]:
    results = {}
    if stats is None: return {}
    for state, samples in sojourns.items():
        arr = np.asarray(samples, dtype=float)
        arr = arr[arr > 0]
        if len(arr) < 5: 
            results[int(state)] = {'params': [1.0, 0, 1.0], 'n': len(arr)} # Default seguro
            continue
        try:
            params = stats.weibull_min.fit(arr, floc=0)
            shape, scale = float(params[0]), float(params[2])
            # Corrección de seguridad
            if scale < 10.0: scale = max(10.0, float(np.mean(arr)))
            results[int(state)] = {'dist': 'weibull_min', 'params': [shape, 0, scale], 'n': len(arr)}
        except:
            results[int(state)] = {'params': [1.0, 0, np.mean(arr)], 'n': len(arr)}
    return results

def save_semi_markov_model(path: str, model: Dict[str, Any]):
    out = {}
    for k, v in model.items():
        if isinstance(v, np.ndarray): out[k] = v.tolist()
        else: out[k] = v
    with open(path, 'w') as fh: json.dump(out, fh, indent=2)

if __name__ == '__main__':
    print(f"--- Entrenando Modelo (Resolución Nativa) ---")
    if not os.path.exists(VITAL_FILENAME):
        print(f"ERROR: No existe {VITAL_FILENAME}"); exit(1)

    vf = vitaldb.VitalFile(VITAL_FILENAME)
    ts, vals = get_icp_signal(vf)
    
    # IMPORTANTE: Verificar que leemos millones de datos, no miles
    print(f"Muestras leídas: {len(vals)}")
    
    mask = (~np.isnan(vals)) & (vals >= 0.0) & (vals <= 100.0)
    vals, ts = vals[mask], ts[mask]

    states = discretize_icp(vals, thresholds=[15.0, 20.0], hysteresis=HYSTERESIS_VAL)
    counts, P = estimate_transition_matrix(states)
    sojourns = extract_sojourns(states, timestamps=ts)
    fits = fit_parametric_sojourns(sojourns)

    model = {
        'P': P, 'counts': counts, 'thresholds': [15.0, 20.0],
        'hysteresis': HYSTERESIS_VAL, 'fits': fits, 'best': fits
    }
    save_semi_markov_model(OUTPUT_FILENAME, model)
    
    print("\n--- Resultados del Ajuste ---")
    for s, d in fits.items():
        if 'params' in d:
            print(f"Estado {s}: Scale = {d['params'][2]:.2f} (n={d.get('n',0)})")