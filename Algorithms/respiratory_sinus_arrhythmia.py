import numpy as np
from scipy.signal import find_peaks
import vitaldb
from compute_rr import compute_rr 

class RespiratorySinusArrhythmia:
    
    def __init__(self,data):
        
        if isinstance(data, vitaldb.VitalFile):
            self._from_vf(data)
        else:
            self._from_df(data)

    def _from_vf(self,vital_file):
        available_tracks = vital_file.get_track_names()
        ecg_track = next(
            (t for t in available_tracks if 'Intellivue/ECG_I' in t),
            next((t for t in available_tracks if 'Intellivue/ECG_II' in t),
                next((t for t in available_tracks if 'Intellivue/ECG_III' in t),
                    next((t for t in available_tracks if 'Intellivue/ECG_V' in t), None))))
        resp_track = next(
            (t for t in available_tracks if 'Intellivue/RESP' in t),
                next((t for t in available_tracks if 'Intellivue/CO2' in t), None))
        
        ecg = vital_file.to_pandas(track_name=ecg_track, interval=1/500)
        resp = vital_file.to_pandas(track_name=resp_track, interval=1/500)

        rr = compute_rr(ecg)
        self.rsa_values = self.compute_rsa(rr, resp)

    def _from_df(self,list_dataframe: list[pd.DataFrame]):
        available_tracks=list_dataframe.keys()
        ecg_track = next(
            (t for t in available_tracks if 'Intellivue/ECG_I' in t),
            next((t for t in available_tracks if 'Intellivue/ECG_II' in t),
                next((t for t in available_tracks if 'Intellivue/ECG_III' in t),
                    next((t for t in available_tracks if 'Intellivue/ECG_V' in t), None))))
        resp_track = next(
            (t for t in available_tracks if 'Intellivue/RESP' in t),
                next((t for t in available_tracks if 'Intellivue/CO2' in t), None))
        ecg = list_dataframe[ecg_track]
        resp = list_dataframe[resp_track]
        rr = compute_rr(ecg)
        self.rsa_values = self.compute_rsa(rr, resp)

    def compute_rsa(self,rr_intervals,resp_signal):
        # Detect peaks in the respiration signal: each peak marks a new respiratory cycle
        peaks, _ = find_peaks(resp_signal, distance=50)

        # Calculate RSA for each respiratory cycle using the peak-to-trough method
        rsa_values = []
        for i in range(len(peaks) - 1):
            start = peaks[i]
            end = peaks[i + 1]
            rr_cycle = rr_intervals[start:end]
            if len(rr_cycle) > 0:
                rsa = np.max(rr_cycle) - np.min(rr_cycle)
                rsa_values.append(rsa)

        # Convert values to float type 
        rsa_values_clean = [float(x) for x in rsa_values]

        return rsa_values_clean
