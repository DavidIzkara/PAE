import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import vitaldb
from util_AL import compute_rr 

class RespiratorySinusArrhythmia:
    
    def __init__(self):
        
        self.last_resp_val = []
        self.last_resp_time = []
        self.last_rr_val = []
        self.last_rr_time = []
        self.values = None

    def compute(self, data):
        if isinstance(data, vitaldb.VitalFile):
            self._from_vf(data)
        else:
            self._from_df(data)
        return self.values
    
    def _from_vf(self,vf):
        # Get all available track names in the VitalFile
        available_tracks = vf.get_track_names()

        # Try to find heart rate wave
        hr_track = next(
            (t for t in available_tracks if 'Intellivue/ECG_I' in t), 
            next((t for t in available_tracks if 'Intellivue/ECG_II' in t),     
                 next((t for t in available_tracks if 'Intellivue/ECG_III' in t), 
                      next((t for t in available_tracks if 'Intellivue/ECG_V' in t), None)))) 

        # Try to find respiratory wave
        resp_track = next(
            (t for t in available_tracks if 'Intellivue/RESP' in t),
            next((t for t in available_tracks if 'Intellivue/CO2' in t), None))
        
        # Convert the signals to NumPy arrays (ECG -> RR)
        hr = vf.to_pandas(track_names=hr_track, interval=1/500, return_timestamp=True)

        rr = compute_rr(hr, hr_track)

        # Convert the signals to NumPy arrays (RESP)
        resp_raw = vf.to_pandas(track_names=resp_track, interval=1/500, return_timestamp=True)
        resp = pd.DataFrame({'value': resp_raw[resp_track], 'Time': resp_raw.index})

        self.values = self.compute_rsa(rr, resp)

    def _from_df(self,list_dataframe: list[pd.DataFrame]):
        available_tracks = list_dataframe.keys()

        # Try to find heart rate wave
        hr_track = next(
            (t for t in available_tracks if 'Intellivue/ECG_I' in t), 
            next((t for t in available_tracks if 'Intellivue/ECG_II' in t),     
                 next((t for t in available_tracks if 'Intellivue/ECG_III' in t), 
                      next((t for t in available_tracks if 'Intellivue/ECG_V' in t), None)))) 
        
        hr_raw = list_dataframe[hr_track]
        hr = pd.DataFrame({hr_track: hr_raw["value"], 'Time': hr_raw["time_ms"] })
        rr = compute_rr(hr, hr_track)

        # Try to find respiratory wave
        resp_track = next(
            (t for t in available_tracks if 'Intellivue/RESP' in t),
            next((t for t in available_tracks if 'Intellivue/CO2' in t), None))

        resp_raw = list_dataframe[resp_track]
        resp = pd.DataFrame({'value': resp_raw["value"], 'Time': resp_raw["time_ms"]})

        self.values = self.compute_rsa(rr, resp)

    def compute_rsa(self, rr_intervals, resp_signal):
        #Extraer datos nuevos a listas
        new_resp_vals = resp_signal['value'].values.tolist()
        new_resp_times = resp_signal['Time'].values.tolist()

        new_rr_vals = rr_intervals['rr'].values.tolist()
        # Usamos Time_fin_ms como referencia para el RR
        new_rr_times = rr_intervals['Time_fin_ms'].values.tolist() 

        #Concatenar con los datos en buffer
        resp_vals = np.array(self.last_resp_val + new_resp_vals)
        resp_times = np.array(self.last_resp_time + new_resp_times)
        
        rr_vals = np.array(self.last_rr_val + new_rr_vals)
        rr_times = np.array(self.last_rr_time + new_rr_times)

        peaks_idx, _ = find_peaks(resp_vals, distance=250) 
        
        results = []
        
        # Necesitamos al menos 2 picos para formar un ciclo cerrado
        if len(peaks_idx) > 1:
            for i in range(len(peaks_idx) - 1):
                # Obtener tiempos de inicio y fin del ciclo respiratorio
                t_start = resp_times[peaks_idx[i]]
                t_end = resp_times[peaks_idx[i+1]]

                idx_start = np.searchsorted(rr_times, t_start)
                idx_end = np.searchsorted(rr_times, t_end)

                rr_cycle = rr_vals[idx_start : idx_end]

                if len(rr_cycle) > 0:
                    rsa_val = np.max(rr_cycle) - np.min(rr_cycle)
                    results.append([t_start, t_end, rsa_val])
            
            # El último pico detectado (peaks_idx[-1]) es el inicio del siguiente ciclo.
            last_peak_idx = peaks_idx[-1]
            last_peak_time = resp_times[last_peak_idx]

            # Desde el último pico hasta el final
            self.last_resp_val = resp_vals[last_peak_idx:].tolist()
            self.last_resp_time = resp_times[last_peak_idx:].tolist()

            # Todos los RRs que ocurren después del último pico respiratorio
            # Usamos searchsorted para encontrar el índice de corte en el array de RR
            cut_rr_idx = np.searchsorted(rr_times, last_peak_time)
            self.last_rr_val = rr_vals[cut_rr_idx:].tolist()
            self.last_rr_time = rr_times[cut_rr_idx:].tolist()

        else:
            # Si no hay suficientes picos (menos de 2), no podemos cerrar ningún ciclo.
            self.last_resp_val = resp_vals.tolist()
            self.last_resp_time = resp_times.tolist()
            self.last_rr_val = rr_vals.tolist()
            self.last_rr_time = rr_times.tolist()

        return pd.DataFrame(results, columns=["Time_ini_ms", "Time_fin_ms", "RSA"])
