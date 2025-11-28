import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import vitaldb
from compute_rr import compute_rr 

class RespiratorySinusArrhythmia:
    
    def __init__(self,data):
        
        if isinstance(data, vitaldb.VitalFile):
            self._from_vf(data)
        else:
            self._from_df(data)

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

        rsa = self.compute_rsa(rr, resp)
        
        values = rsa

        self.values = pd.DataFrame({'Time_ini_ms': values["Time_ini_ms"], 'Time_fin_ms': values["Time_fin_ms"], 'RSA': values["RSA"]})

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

        # Calculate RSA metrics
        rsa = self.compute_rsa(rr, resp)

        values = rsa

        self.values = pd.DataFrame({'Time_ini_ms': values["Time_ini_ms"], 'Time_fin_ms': values["Time_fin_ms"], 'RSA': values["RSA"]})

    def compute_rsa(self,rr_intervals,resp_signal):
        resp_vals = resp_df['value'].values
        resp_times = resp_df['Time'].values

        rr_vals = rr_df['rr'].values
        rr_times = rr_df['Time_fin_ms'].values 

        peaks_idx, _ = find_peaks(resp_vals, distance=250) 
        
        results = []

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

        return pd.DataFrame(results, columns=["Time_ini_ms", "Time_fin_ms", "RSA"])
