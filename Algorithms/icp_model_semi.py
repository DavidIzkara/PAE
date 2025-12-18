import numpy as np
import json
import os
import vitaldb
import argparse
import glob
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
    """Discretiza valores ICP en 3 estados usando histéresis (VECTORIZADO)."""
    values = np.asarray(values, dtype=float).ravel()
    states = np.zeros_like(values, dtype=int)
    if len(values) == 0: return states

    t1, t2 = thresholds[0], thresholds[1]
    t1_low, t1_high = t1 - hysteresis, t1 + hysteresis
    t2_low, t2_high = t2 - hysteresis, t2 + hysteresis
    
    # Start with simple thresholding to get initial state
    current_state = 0
    if values[0] >= t2: current_state = 2
    elif values[0] >= t1: current_state = 1
    states[0] = current_state
    
    # Process transitions with hysteresis via state machine
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
    """Estimate transition matrix using fully vectorized operations."""
    valid_mask = ~np.isnan(states)
    states = np.asarray(states, dtype=int)[valid_mask]
    n_states = int(np.max(states)) + 1 if len(states) > 0 else 3
    
    # Vectorized transition counting using np.add.at
    from_states = states[:-1]
    to_states = states[1:]
    
    # Create 1D index into 2D counts array: counts[i,j] -> counts_flat[i*n_states + j]
    indices = from_states * n_states + to_states
    counts_flat = np.zeros(n_states * n_states, dtype=int)
    np.add.at(counts_flat, indices, 1)
    counts = counts_flat.reshape((n_states, n_states))
    
    # Normalize to transition matrix
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
    parser = argparse.ArgumentParser(description='Estimate semi-Markov model from ICP vital signals')
    parser.add_argument('--input-file', type=str, default=None, help='Single vital file to process')
    parser.add_argument('--input-dir', type=str, default=None, help='Directory with multiple vital files for batch aggregation')
    parser.add_argument('--output-file', type=str, default=None, help='Output JSON model file')
    parser.add_argument('--hysteresis', type=float, default=HYSTERESIS_VAL, help='Hysteresis value for discretization')
    args = parser.parse_args()
    
    # Default behavior: if no arguments provided, use batch mode on TrainingData
    if args.input_file is None and args.input_dir is None:
        args.input_dir = 'TrainingData'
        if args.output_file is None:
            args.output_file = 'semi_markov_model_aggregated.json'
    elif args.output_file is None:
        args.output_file = OUTPUT_FILENAME
    
    # Determine mode: single file or batch directory
    if args.input_dir:
        print(f"--- BATCH MODE: Processing directory {args.input_dir} ---")
        vital_files = sorted(glob.glob(os.path.join(args.input_dir, '*.vital')))
        if not vital_files:
            print(f"ERROR: No .vital files found in {args.input_dir}")
            exit(1)
        print(f"Found {len(vital_files)} vital files")
        
        # Aggregate counts and sojourns across all files
        aggregated_counts = None
        aggregated_sojourns = defaultdict(list)
        file_count = 0
        
        for vital_file in vital_files:
            try:
                print(f"Processing: {os.path.basename(vital_file)}", end=" ... ")
                vf = vitaldb.VitalFile(vital_file)
                ts, vals = get_icp_signal(vf)
                
                # Validate and filter
                mask = (~np.isnan(vals)) & (vals >= 0.0) & (vals <= 100.0)
                vals, ts = vals[mask], ts[mask]
                
                if len(vals) == 0:
                    print("SKIP (no valid data)")
                    continue
                
                # Process this file
                states = discretize_icp(vals, thresholds=[15.0, 20.0], hysteresis=args.hysteresis)
                counts, P = estimate_transition_matrix(states)
                sojourns = extract_sojourns(states, timestamps=ts)
                
                # Aggregate
                if aggregated_counts is None:
                    aggregated_counts = counts.copy()
                else:
                    aggregated_counts += counts
                
                for state, durations in sojourns.items():
                    aggregated_sojourns[state].extend(durations)
                
                file_count += 1
                print(f"OK ({len(vals)} samples, {counts.sum()} transitions)")
            except Exception as e:
                print(f"ERROR: {e}")
        
        if file_count == 0:
            print("ERROR: No files processed successfully")
            exit(1)
        
        print(f"\n--- Aggregating {file_count} files ---")
        
        # Normalize aggregated counts to transition matrix
        row_sums = aggregated_counts.sum(axis=1, keepdims=True).astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            P = np.divide(aggregated_counts, row_sums, where=(row_sums != 0))
        P[np.isnan(P)] = 0.0
        
        # Fit on aggregated sojourns
        fits = fit_parametric_sojourns(dict(aggregated_sojourns))
        
        model = {
            'P': P, 'counts': aggregated_counts, 'thresholds': [15.0, 20.0],
            'hysteresis': args.hysteresis, 'fits': fits, 'best': fits
        }
        save_semi_markov_model(args.output_file, model)
        
        print(f"\nAggregated model saved to {args.output_file}")
        print("\n--- Aggregated Model Summary ---")
        print(f"Total transitions: {aggregated_counts.sum()}")
        print(f"Transition matrix P:\n{P}")
        for s, d in fits.items():
            if 'params' in d:
                print(f"State {s}: Scale = {d['params'][2]:.2f} (n={d.get('n',0)} sojourns)")
    
    else:
        # Single file mode
        input_file = args.input_file if args.input_file else VITAL_FILENAME
        output_file = args.output_file
        
        print(f"--- Single File Mode: Processing {input_file} ---")
        if not os.path.exists(input_file):
            print(f"ERROR: No existe {input_file}"); exit(1)

        vf = vitaldb.VitalFile(input_file)
        ts, vals = get_icp_signal(vf)
        
        # IMPORTANTE: Verificar que leemos millones de datos, no miles
        print(f"Muestras leídas: {len(vals)}")
        
        mask = (~np.isnan(vals)) & (vals >= 0.0) & (vals <= 100.0)
        vals, ts = vals[mask], ts[mask]

        states = discretize_icp(vals, thresholds=[15.0, 20.0], hysteresis=args.hysteresis)
        counts, P = estimate_transition_matrix(states)
        sojourns = extract_sojourns(states, timestamps=ts)
        fits = fit_parametric_sojourns(sojourns)

        model = {
            'P': P, 'counts': counts, 'thresholds': [15.0, 20.0],
            'hysteresis': args.hysteresis, 'fits': fits, 'best': fits
        }
        save_semi_markov_model(output_file, model)
        
        print("\n--- Resultados del Ajuste ---")
        for s, d in fits.items():
            if 'params' in d:
                print(f"Estado {s}: Scale = {d['params'][2]:.2f} (n={d.get('n',0)})")
