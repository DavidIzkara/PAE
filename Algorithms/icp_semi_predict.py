import numpy as np
from scipy.stats import exponweib
import vitaldb
import os
import json
import argparse
import time

# === CONFIGURACIÓN ===
VITAL_PATH = 'D:/UPC/CUATRI/PAE/data_icp_181125/data/kuigebjtu_250531_224330.vital'

# === FUNCIONES AUXILIARES ===

def discretize_icp(icp_values, thresholds=None, hysteresis=0.5):
    """Discretiza valores ICP en 3 estados usando dos thresholds y histéresis."""
    if thresholds is None:
        t1, t2 = 15.0, 20.0
    else:
        t1, t2 = float(thresholds[0]), float(thresholds[1])
        
    icp = np.asarray(icp_values, dtype=float).ravel()
    states = np.zeros_like(icp, dtype=int)
    
    if len(icp) == 0:
        return states

    # Estado inicial
    current_state = 0
    if icp[0] >= t2: current_state = 2
    elif icp[0] >= t1: current_state = 1
    states[0] = current_state
    
    # Pre-calculamos límites
    t1_low, t1_high = t1 - hysteresis, t1 + hysteresis
    t2_low, t2_high = t2 - hysteresis, t2 + hysteresis

    for i in range(1, len(icp)):
        val = icp[i]
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

def load_semi_markov_model(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            obj = json.load(f)
        if 'P' in obj:
            obj['P'] = np.asarray(obj['P'], dtype=float)
        return obj
    except Exception as e:
        print(f"Failed to load semi-Markov model from {path}: {e}")
        return {}

# === FUNCIONES DE VECTORIZACIÓN ===

def get_vectorized_durations(states, timestamps):
    """Calcula la duración acumulada actual para cada punto de manera vectorizada."""
    n = len(states)
    if n == 0: return np.array([])
    
    # Índices donde cambia el estado
    change_mask = states[:-1] != states[1:]
    change_indices = np.flatnonzero(change_mask) + 1
    
    # Añadimos inicio y fin
    run_starts = np.concatenate(([0], change_indices, [n]))
    
    durations = np.zeros(n, dtype=float)
    
    # Llenamos duraciones por bloques
    for start, end in zip(run_starts[:-1], run_starts[1:]):
        durations[start:end] = timestamps[start:end] - timestamps[start]
        
    return durations

def get_ground_truth_vectorized(states, timestamps, horizon):
    """Determina si habrá un cambio de estado dentro del horizonte."""
    n = len(states)
    if n == 0: return np.zeros(0, dtype=bool)
    
    change_mask = states[:-1] != states[1:]
    change_indices = np.flatnonzero(change_mask) + 1
    
    if len(change_indices) == 0:
        return np.zeros(n, dtype=bool)
    
    # Mapear cada punto a su siguiente índice de cambio
    next_change_idx_map = np.searchsorted(change_indices, np.arange(n), side='right')
    valid_mask = next_change_idx_map < len(change_indices)
    
    has_changed = np.zeros(n, dtype=bool)
    
    indices_of_next_change = change_indices[next_change_idx_map[valid_mask]]
    times_of_next_change = timestamps[indices_of_next_change]
    time_remaining = times_of_next_change - timestamps[valid_mask]
    
    has_changed[valid_mask] = time_remaining <= horizon
    
    return has_changed

def predict_risk_vectorized(states, durations, best_fits, horizon):
    """Calcula probabilidad de riesgo vectorizada."""
    n = len(states)
    risk_probs = np.zeros(n, dtype=float)
    unique_states = np.unique(states)
    
    for s in unique_states:
        if s not in best_fits:
            continue
            
        shape, scale = best_fits[s]
        mask = (states == s)
        durs = durations[mask]
        
        # Weibull Survival
        # S(t) = exp(-(t/scale)^shape)
        # Prob. Cambio = 1 - S(t+H)/S(t)
        
        # Evitar divisiones por cero si scale es mala
        scale = max(1e-3, scale)
        
        t_norm = durs / scale
        t_fut_norm = (durs + horizon) / scale
        
        surv_now = np.exp(-(t_norm ** shape))
        surv_future = np.exp(-(t_fut_norm ** shape))
        
        # Máscara segura para división
        safe_div = surv_now > 1e-9
        
        p_change = np.ones_like(durs) # Default 1.0 (cambio seguro) si ya debió cambiar
        p_change[safe_div] = 1.0 - (surv_future[safe_div] / surv_now[safe_div])
        
        risk_probs[mask] = np.clip(p_change, 0.0, 1.0)
        
    return risk_probs

# === MAIN ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-Markov ICP prediction (Optimized + Debug).")
    parser.add_argument('--vital', type=str, default=VITAL_PATH, help='Path to .vital file')
    parser.add_argument('--model-file', type=str, default='semi_markov_model.json', help='Model JSON')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit samples')
    parser.add_argument('--horizon', type=float, default=60.0, help='Horizonte (segundos)')
    parser.add_argument('--alert-threshold', type=float, default=0.5, help='Umbral Alerta (0.0-1.0)')
    parser.add_argument('--hysteresis', type=float, default=0.5, help='Histéresis (mmHg)')
    
    args = parser.parse_args()

    print("=== Testing Semi-Markov ICP Model (Optimized) ===")
    t0 = time.time()

    # 1. Cargar Modelo
    model = load_semi_markov_model(args.model_file)
    if not model:
        print(f"Error: No se encontró el modelo en {args.model_file}")
        raise SystemExit(1)

    # 2. Cargar Datos
    rec = vitaldb.VitalFile(args.vital if args.vital is not None else VITAL_PATH)
    icp, timestamps = None, None
    try:
        res = rec.to_numpy('ICP', 0, return_timestamp=True)
        if isinstance(res, tuple) and len(res) >= 2:
            timestamps, icp = np.asarray(res[0]).ravel(), np.asarray(res[1]).ravel()
        else:
            arr = np.asarray(res)
            if arr.ndim == 2 and arr.shape[1] >= 2: timestamps, icp = arr[:, 0], arr[:, 1]
            else: icp = arr.ravel()
    except:
        try: icp = np.asarray(rec.to_numpy('ICP', 0)).ravel()
        except: pass

    if icp is None or icp.size == 0: raise ValueError("No ICP data.")
    if timestamps is None: timestamps = np.arange(len(icp), dtype=float)

    # Limpieza
    mask = (~np.isnan(icp)) & (icp >= 0.0) & (icp <= 50.0)
    icp, timestamps = icp[mask], timestamps[mask]
    if args.max_samples and icp.size > args.max_samples:
        icp, timestamps = icp[:args.max_samples], timestamps[:args.max_samples]

    print(f"Datos cargados: {len(icp)} muestras en {time.time()-t0:.2f}s")

    # 3. Parsear Fits (Debug Step 1)
    raw_fits = model.get('best') or model.get('fits') or {}
    best_fits = {}
    
    print("\n--- DIAGNÓSTICO 1: Parámetros Weibull Cargados ---")
    if isinstance(raw_fits, dict):
        for k, v in raw_fits.items():
            try:
                idx = int(k)
                shape, scale = 1.0, 1.0
                if isinstance(v, dict) and 'params' in v:
                    p = v['params']
                    shape, scale = (float(p[0]), float(p[-1])) if len(p) >= 3 else (float(p[0]), float(p[1]))
                elif isinstance(v, (list, tuple)) and len(v) >= 2:
                    shape, scale = float(v[0]), float(v[1])
                
                best_fits[idx] = (shape, scale)
                print(f"  Estado {idx}: Shape={shape:.2f}, Scale={scale:.2f}")
                
                if scale < 10.0:
                    print(f"    [!] ALERTA: La escala del Estado {idx} es muy pequeña ({scale:.2f}s).")
                    print(f"        Esto hará que la probabilidad de cambio sea 100% casi siempre.")
            except: continue
    else:
        print("  [!] No se encontraron fits válidos en el JSON.")

    # 4. Discretización y Duraciones (Debug Step 4)
    thresholds = [float(t) for t in model.get('thresholds', [15.0, 20.0])]
    states = discretize_icp(icp, thresholds=thresholds, hysteresis=args.hysteresis)
    durations = get_vectorized_durations(states, timestamps)
    
    mean_dur = np.mean(durations) if len(durations) > 0 else 0.0
    print("\n--- DIAGNÓSTICO 2: Duración de Estados ---")
    print(f"  Duración Media de un estado: {mean_dur:.2f} segundos")
    if mean_dur < 10.0:
        print(f"  [!] ALERTA: Los estados duran muy poco (ruido).")
        print(f"      Acción recomendada: Aumentar --hysteresis (actual: {args.hysteresis})")

    # 5. Evaluación Vectorizada
    print(f"\n--- Ejecutando Predicción (Horizonte={args.horizon}s, Umbral={args.alert_threshold}) ---")
    
    t_eval = time.time()
    has_changed = get_ground_truth_vectorized(states, timestamps, args.horizon)
    risk_probs = predict_risk_vectorized(states, durations, best_fits, args.horizon)
    predicted = risk_probs > args.alert_threshold
    
    # Métricas
    tp = np.sum(predicted & has_changed)
    fp = np.sum(predicted & (~has_changed))
    fn = np.sum((~predicted) & has_changed)
    tn = np.sum((~predicted) & (~has_changed))
    
    elapsed = time.time() - t_eval
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Evaluación terminada en {elapsed:.2f}s\n")
    print("="*45)
    print(f"RESULTADOS DE PREDICCIÓN")
    print("="*45)
    print(f"Muestras Totales:     {len(icp)}")
    print(f"Alertas Lanzadas:     {tp + fp} ({(tp+fp)/len(icp)*100:.1f}%)")
    print(f"Cambios Reales:       {tp + fn} ({(tp+fn)/len(icp)*100:.1f}%)")
    print("-" * 45)
    print(f"TP: {tp:<8} FP: {fp:<8}")
    print(f"FN: {fn:<8} TN: {tn:<8}")
    print("-" * 45)
    print(f"PRECISION:   {precision:.4f}  (¿Es la alerta fiable?)")
    print(f"RECALL:      {recall:.4f}     (¿Detectamos los cambios?)")
    print(f"SPECIFICITY: {spec:.4f}      (¿Evitamos falsas alarmas?)")
    print(f"F1-SCORE:    {f1:.4f}")
    print("="*45)