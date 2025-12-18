import numpy as np
from scipy.stats import exponweib
import vitaldb
import os
import json
import argparse
import time
import subprocess
import glob
import sys
import re

# === CONFIGURACIÓN ===
VITAL_PATH = 'kuigebjtu_250531_224330.vital'

# === FUNCIONES AUXILIARES ===

def discretize_icp(icp_values, thresholds=None, hysteresis=1.0):
    if thresholds is None: t1, t2 = 15.0, 20.0
    else: t1, t2 = float(thresholds[0]), float(thresholds[1])
        
    icp = np.asarray(icp_values, dtype=float).ravel()
    states = np.zeros_like(icp, dtype=int)
    if len(icp) == 0: return states

    current_state = 0
    if icp[0] >= t2: current_state = 2
    elif icp[0] >= t1: current_state = 1
    states[0] = current_state
    
    t1_low, t1_high = t1 - hysteresis, t1 + hysteresis
    t2_low, t2_high = t2 - hysteresis, t2 + hysteresis

    for i in range(1, len(icp)):
        val = icp[i]
        if current_state == 0:
            if val >= t1_high: current_state = 1; 
            if val >= t2_high: current_state = 2
        elif current_state == 1:
            if val <= t1_low: current_state = 0
            elif val >= t2_high: current_state = 2
        elif current_state == 2:
            if val <= t2_low: current_state = 1; 
            if val <= t1_low: current_state = 0
        states[i] = current_state
    return states

def load_semi_markov_model(path: str) -> dict:
    if not os.path.exists(path): return {}
    with open(path, 'r') as f:
        obj = json.load(f)
    if 'P' in obj: obj['P'] = np.asarray(obj['P'], dtype=float)
    return obj

def get_vectorized_durations(states, timestamps):
    n = len(states)
    if n == 0: return np.array([])
    change_mask = states[:-1] != states[1:]
    change_indices = np.flatnonzero(change_mask) + 1
    run_starts = np.concatenate(([0], change_indices, [n]))
    durations = np.zeros(n, dtype=float)
    for start, end in zip(run_starts[:-1], run_starts[1:]):
        durations[start:end] = timestamps[start:end] - timestamps[start]
    return durations

# === LÓGICA DE CRITICIDAD Y HORIZONTE ADAPTATIVO ===

def get_adaptive_horizons(states):
    """
    Define el horizonte según el estado clínico.
    Estado 0 (Estable): 300s (5 min) - Evitar alarmas prematuras.
    Estado 1 (Alerta):  60s  (1 min) - Sensibilidad máxima.
    Estado 2 (Crisis):  60s
    """
    horizons = np.zeros_like(states, dtype=float)
    horizons[states == 0] = 300.0
    horizons[states == 1] = 60.0
    horizons[states == 2] = 60.0
    return horizons

def get_critical_ground_truth(states, timestamps, horizons_array):
    """
    Ground Truth inteligente:
    1. Solo es TRUE si hay cambio dentro del horizonte específico de ese punto.
    2. Solo es TRUE si el cambio es A PEOR (Empeoramiento).
    """
    n = len(states)
    if n == 0: return np.zeros(0, dtype=bool)
    
    change_mask = states[:-1] != states[1:]
    change_indices = np.flatnonzero(change_mask) + 1
    
    if len(change_indices) == 0: return np.zeros(n, dtype=bool)
    
    next_change_idx_map = np.searchsorted(change_indices, np.arange(n), side='right')
    valid_mask = next_change_idx_map < len(change_indices)
    
    has_changed_critically = np.zeros(n, dtype=bool)
    
    # Índices del siguiente cambio
    indices_of_next_change = change_indices[next_change_idx_map[valid_mask]]
    
    # A) Chequeo de Tiempo (con horizonte adaptativo por punto)
    time_remaining = timestamps[indices_of_next_change] - timestamps[valid_mask]
    is_in_time = time_remaining <= horizons_array[valid_mask]
    
    # B) Chequeo de Criticidad (¿Es un empeoramiento?)
    # Comparar estado actual vs estado siguiente
    current_s = states[valid_mask]
    next_s = states[indices_of_next_change]
    is_worsening = next_s > current_s
    
    # Combinar: A tiempo AND Empeora
    has_changed_critically[valid_mask] = is_in_time & is_worsening
    
    return has_changed_critically

def predict_critical_risk(states, durations, best_fits, P_matrix, horizons_array):
    """
    Calcula riesgo ponderado por gravedad.
    Riesgo = P(Salir en H) * P(Ir a Peor)
    """
    n = len(states)
    risk_probs = np.zeros(n, dtype=float)
    unique_states = np.unique(states)
    
    for s in unique_states:
        if s not in best_fits: continue
        shape, scale = best_fits[s]
        scale = max(1e-3, scale)
        
        mask = (states == s)
        durs = durations[mask]
        hs = horizons_array[mask] # Horizontes correspondientes
        
        # 1. Probabilidad de Salir (Weibull)
        t_norm = durs / scale
        t_fut_norm = (durs + hs) / scale
        
        surv_now = np.exp(-(t_norm ** shape))
        surv_future = np.exp(-(t_fut_norm ** shape))
        
        safe_div = surv_now > 1e-9
        p_leave = np.zeros_like(durs)
        p_leave[safe_div] = 1.0 - (surv_future[safe_div] / surv_now[safe_div])
        p_leave[~safe_div] = 1.0
        
        # 2. Factor de Gravedad (Probabilidad de Empeorar)
        # Extraemos de la matriz P la suma de prob hacia estados mayores
        # P shape: (3,3). Row s.
        # Worsening probability = sum(P[s, next_s]) where next_s > s
        if P_matrix is not None and s < (P_matrix.shape[0] - 1):
            worsening_prob = np.sum(P_matrix[s, s+1:])
        else:
            worsening_prob = 0.0 # Si estás en el último estado, no puedes empeorar
            
        # Riesgo Final
        risk_probs[mask] = np.clip(p_leave * worsening_prob, 0.0, 1.0)
        
    return risk_probs

# === MAIN ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vital', type=str, default=None)
    parser.add_argument('--model-file', type=str, default='semi_markov_model_aggregated.json')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--input-dir', type=str, default=None)
    # Nota: horizon ahora es adaptativo interno, este argumento es solo legacy/default
    parser.add_argument('--alert-threshold', type=float, default=0.05, help='OJO: Al ponderar, los valores bajan mucho. Usar < 0.1')
    parser.add_argument('--hysteresis', type=float, default=1.0)
    args = parser.parse_args()

    # Modo Batch
    if args.input_dir is None and not args.vital:
        args.input_dir = 'PredictionGroup'

    if args.input_dir:
        os.makedirs(args.input_dir, exist_ok=True)
        vital_files = sorted(glob.glob(os.path.join(args.input_dir, '*.vital')))
        print(f"--- BATCH PREDICTION (ADAPTIVE + CRITICAL): {len(vital_files)} files ---")
        
        agg = {'TP':0,'FP':0,'FN':0,'TN':0,'samples':0}
        per_file = {}
        
        for vf in vital_files:
            # Recursión al mismo script para procesar archivo
            cmd = [sys.executable, os.path.abspath(__file__), '--vital', vf, '--model-file', args.model_file, 
                   '--alert-threshold', str(args.alert_threshold), '--hysteresis', str(args.hysteresis)]
            if args.max_samples: cmd += ['--max-samples', str(args.max_samples)]
            
            print(f'Running: {os.path.basename(vf)} ...', end=' ', flush=True)
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                out = proc.stdout
                
                # Parsear output
                m = re.search(r'TP:\s*(\d+)\s*FP:\s*(\d+).*?FN:\s*(\d+)\s*TN:\s*(\d+)', out, re.S)
                if m:
                    tp,fp,fn,tn = map(int, m.groups())
                    print(f"OK (TP={tp} FP={fp})")
                    agg['TP']+=tp; agg['FP']+=fp; agg['FN']+=fn; agg['TN']+=tn
                    # Calcular muestras total aprox
                    s = tp+fp+fn+tn
                    agg['samples']+=s
                else:
                    print("ERROR parsing output")
            except Exception as e:
                print(f"FAIL: {e}")

        # Métricas Globales
        TP,FP,FN,TN = agg['TP'], agg['FP'], agg['FN'], agg['TN']
        prec = TP/(TP+FP) if (TP+FP)>0 else 0
        rec = TP/(TP+FN) if (TP+FN)>0 else 0
        spec = TN/(TN+FP) if (TN+FP)>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        
        print('\n=== BATCH AGGREGATE RESULTS (ADAPTIVE) ===')
        print(f'TP: {TP}  FP: {FP}  FN: {FN}  TN: {TN}')
        print(f'Precision:   {prec:.4f}')
        print(f'Recall:      {rec:.4f}')
        print(f'Specificity: {spec:.4f}')
        print(f'F1 Score:    {f1:.4f}')
        exit(0)

    # Modo Single File
    model = load_semi_markov_model(args.model_file)
    if not model: exit(1)

    rec = vitaldb.VitalFile(args.vital if args.vital else VITAL_PATH)
    try: vals = rec.to_numpy('ICP', 0); ts = np.arange(len(vals), dtype=float)
    except: vals = rec.to_numpy('Intellivue/ICP', 0); ts = np.arange(len(vals), dtype=float)
    vals = np.asarray(vals).ravel()
    
    mask = (~np.isnan(vals)) & (vals >= 0.0) & (vals <= 100.0)
    vals, ts = vals[mask], ts[mask]
    if args.max_samples: vals, ts = vals[:args.max_samples], ts[:args.max_samples]

    # Preparar datos
    thresholds = [float(t) for t in model.get('thresholds', [15.0, 20.0])]
    states = discretize_icp(vals, thresholds, args.hysteresis)
    durations = get_vectorized_durations(states, ts)
    
    # Cargar fits y P
    raw_fits = model.get('best') or model.get('fits') or {}
    best_fits = {}
    if isinstance(raw_fits, dict):
        for k, v in raw_fits.items():
            try:
                p = v.get('params', [])
                if len(p) >= 3: best_fits[int(k)] = (float(p[0]), float(p[-1]))
            except: pass
    P_matrix = model.get('P')

    # === NUEVA LÓGICA ===
    # 1. Definir Horizontes Adaptativos
    horizons = get_adaptive_horizons(states)
    
    # 2. Calcular Ground Truth Crítico (Solo empeoramiento)
    has_changed = get_critical_ground_truth(states, ts, horizons)
    
    # 3. Calcular Riesgo Ponderado (Weibull * P_worsening)
    risk_probs = predict_critical_risk(states, durations, best_fits, P_matrix, horizons)
    
    # Evaluar
    predicted = risk_probs > args.alert_threshold
    
    tp = np.sum(predicted & has_changed)
    fp = np.sum(predicted & (~has_changed))
    fn = np.sum((~predicted) & has_changed)
    tn = np.sum((~predicted) & (~has_changed))
    
    print(f"\nRESULTADOS SINGLE FILE (Adaptativo):")
    print(f"TP: {tp:<8} FP: {fp:<8} FN: {fn:<8} TN: {tn:<8}")