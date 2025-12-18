import time
import json
import os
import numpy as np
import vitaldb
import argparse
import sys
from collections import deque

class ICPMonitor:
    def __init__(self, model_path, alert_threshold, hysteresis=1.0):
        self.base_threshold = alert_threshold
        self.hysteresis = hysteresis
        
        # Cargar Modelo JSON
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
            
        with open(model_path, 'r') as f:
            self.model = json.load(f)
        
        self.P = np.array(self.model['P'])
        self.thresholds = self.model['thresholds'] # [15.0, 20.0]
        
        # Parsear Fits
        self.fits = {}
        raw_fits = self.model.get('best') or self.model.get('fits') or {}
        for k, v in raw_fits.items():
            if 'params' in v:
                p = v['params']
                self.fits[int(k)] = (float(p[0]), float(p[-1]))

        # Variables de Estado
        self.current_state = 0
        self.start_time_of_state = 0.0
        self.last_ts = None
        self.duration = 0.0
        
        # Historial para tendencia (últimos ~15 segundos si es 1Hz)
        self.history = deque(maxlen=15) 

    def _discretize_step(self, val):
        """Aplica histéresis punto a punto."""
        t1, t2 = self.thresholds
        h = self.hysteresis
        
        if self.current_state == 0:
            if val >= (t1 + h): 
                return 1 if val < (t2 - h) else 2
        elif self.current_state == 1:
            if val <= (t1 - h): return 0
            if val >= (t2 + h): return 2
        elif self.current_state == 2:
            if val <= (t2 - h): 
                return 1 if val > (t1 - h) else 0
                
        return self.current_state

    def _get_adaptive_horizon(self):
        if self.current_state == 0: return 300.0 # 5 min (Estable)
        return 60.0 # 1 min (Alerta/Crisis)

    def _get_contextual_threshold(self):
        if self.current_state == 0:
            # Estado 0: Exigimos una señal muy fuerte para pitar (evitar ruido basal)
            return max(self.base_threshold * 2.5, 0.025)
        else:
            # Estado 1/2: Máxima sensibilidad
            return self.base_threshold

    def _calculate_trend(self):
        """Calcula la pendiente (mmHg/s) de forma robusta."""
        n = len(self.history)
        if n < 5: return 0.0
        
        # Convertir a numpy para cálculo
        data = np.array(self.history)
        t = data[:, 0]
        y = data[:, 1]
        
        # Normalizar tiempo (t relativo) para estabilidad numérica
        t = t - t[0]
        
        var_t = np.var(t)
        if var_t < 1e-6: # Protección contra división por cero
            return 0.0
            
        # Pendiente = Covarianza / Varianza
        slope = np.cov(t, y)[0, 1] / var_t
        return slope

    def update(self, val, ts):
        # 1. Historia y Tendencia
        self.history.append((ts, val))
        trend = self._calculate_trend()

        # 2. Inicialización
        if self.last_ts is None:
            self.last_ts = ts
            self.start_time_of_state = ts
            if val >= self.thresholds[1]: self.current_state = 2
            elif val >= self.thresholds[0]: self.current_state = 1
            else: self.current_state = 0
            return 0.0, False, self.current_state, trend

        # 3. Estado y Duración
        new_state = self._discretize_step(val)
        if new_state != self.current_state:
            self.current_state = new_state
            self.start_time_of_state = ts
            self.duration = 0.0
        else:
            self.duration = ts - self.start_time_of_state
        self.last_ts = ts
        
        # 4. Cálculo Riesgo Weibull
        if self.current_state not in self.fits:
            return 0.0, False, self.current_state, trend

        shape, scale = self.fits[self.current_state]
        scale = max(1e-3, scale)
        horizon = self._get_adaptive_horizon()
        
        # Log-Space Hazard
        log_now = -((self.duration / scale) ** shape)
        log_fut = -(((self.duration + horizon) / scale) ** shape)
        prob_leave = 1.0 - np.exp(log_fut - log_now)
        
        worsening_prob = 0.0
        if self.current_state < 2:
            row = self.P[self.current_state]
            worsening_prob = np.sum(row[self.current_state+1:])
            
        # --- CORRECCIÓN AQUÍ: Usar prob_leave ---
        risk = prob_leave * worsening_prob
        
        # 5. DECISIÓN DE ALERTA
        threshold = self._get_contextual_threshold()
        is_alert = risk > threshold
        
        # === GUARDIAS CLÍNICAS (Safety Guards) ===
        
        # A. Silencio Absoluto en zona segura (<12 mmHg)
        if val < 12.0:
            is_alert = False

        # B. FILTRO DE ESTABILIDAD
        # Si estamos en Estado 1 (Alerta) pero la tendencia es:
        # - Negativa (Recuperando)
        # - O muy baja positiva (Estable/Plana, < 0.015 mmHg/s)
        # ENTONCES SILENCIAMOS.
        if self.current_state == 1 and trend < 0.015:
            is_alert = False

        # C. ALARMA DE EMERGENCIA (>20 mmHg)
        # Aquí no hay discusión. Si pasa de 20, pita siempre.
        if val >= 20.0:
            is_alert = True
            risk = max(risk, 1.0) # Visualmente riesgo máximo

        return risk, is_alert, self.current_state, trend

# === SIMULACIÓN ===

def run_simulation(vital_file, model_file, threshold):
    print(f"--- MONITOR ICP (Producción Final) ---")
    print(f"Paciente: {os.path.basename(vital_file)}")
    print(f"Umbral Base: {threshold}")
    print(f"Estrategia: Híbrida (IA + Tendencia + Emergencia)")
    
    vf = vitaldb.VitalFile(vital_file)
    try: vals = vf.to_numpy('ICP', interval=0) # Carga raw para máxima precisión
    except: vals = vf.to_numpy('Intellivue/ICP', interval=0)
    vals = np.asarray(vals).ravel()
    ts = np.arange(len(vals), dtype=float) 
    
    mask = (~np.isnan(vals)) & (vals >= 0.0) & (vals <= 100.0)
    vals = vals[mask]
    ts = ts[mask]
    
    monitor = ICPMonitor(model_file, alert_threshold=threshold, hysteresis=1.0)
    
    print("\n[TIEMPO] | [ICP]  | [EST] | [TREND] | [RIESGO] | [ACCION]")
    print("-" * 75)
    
    alerts_triggered = 0
    
    for i, (v, t) in enumerate(zip(vals, ts)):
        risk, alert, state, trend = monitor.update(v, t)
        
        # Lógica de visualización en consola
        if alert:
            alerts_triggered += 1
            print(f"{t:8.1f} | {v:5.1f} |  {state}  | {trend:6.3f}  | {risk:.4f}   | !!! ALERTA !!!")
        
        # Mostrar estado de vigilancia en zona gris (12-20) sin alerta
        elif v >= 12.0 and i % 100 == 0:
            status = "[Recuperando]" if (state==1 and trend < 0.015) else "[Vigilando...]"
            print(f"{t:8.1f} | {v:5.1f} |  {state}  | {trend:6.3f}  | {risk:.4f}   | {status}")
            
    print("-" * 75)
    print(f"Total Alertas Reales: {alerts_triggered}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vital', type=str, required=True)
    parser.add_argument('--model', type=str, default='semi_markov_model_aggregated.json')
    parser.add_argument('--threshold', type=float, default=0.0075)
    args = parser.parse_args()
    
    run_simulation(args.vital, args.model, args.threshold)