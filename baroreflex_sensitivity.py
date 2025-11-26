import vitaldb
import numpy as np
from scipy.stats import linregress
from scipy.signal import find_peaks
from compute_rr import compute_rr

class BaroreflexSensitivity:
    
    def __init__(self,data):
        
        if isinstance(data, vitaldb.VitalFile):
            self._from_vf(data)
        else:
            self._from_df(data)

    def _from_vf(self,vf):
            # Get all available track names in the VitalFile
            available_tracks= vf.get_track_names()

            #Try to find heart rate wave
            ecg_track = next(
            (t for t in available_tracks if 'Intellivue/ECG_I' in t), 
            next((t for t in available_tracks if 'Intellivue/ECG_II' in t),     
                 next((t for t in available_tracks if 'Intellivue/ECG_III' in t), 
                      next((t for t in available_tracks if 'Intellivue/ECG_V' in t), None)))) 

            ecg=vf.to_pandas(track_name=ecg_track,interval=1/500)
            rr = compute_rr(ecg)

            art_track = next(
                    (t for t in available_tracks if 'Intellivue/ART' in t), 
                        next((t for t in available_tracks if 'Intellivue/ABP' in t), None))
            
            art=vf.to_pandas(track_name=art_track,interval=1/500)
            sbp = self.compute_sbp(art)

            #Calculate BRS
            self.brs = self.compute_brs(sbp,rr)

    def _from_df(self,list_dataframe: list [pd.DataFrame]):
            available_tracks = list_dataframe.keys()
            
            #Try to find heart rate wave
            ecg_track = next(
            (t for t in available_tracks if 'Intellivue/ECG_I' in t), 
            next((t for t in available_tracks if 'Intellivue/ECG_II' in t),     
                 next((t for t in available_tracks if 'Intellivue/ECG_III' in t), 
                      next((t for t in available_tracks if 'Intellivue/ECG_V' in t), None)))) 
            
            ecg = list_dataframe[ecg_track]
            rr = compute_rr(ecg)

            art_track = next(
                    (t for t in available_tracks if 'Intellivue/ART' in t), 
                        next((t for t in available_tracks if 'Intellivue/ABP' in t), None))
            art = list_dataframe[art_track]
            sbp = self.compute_sbp(art)
            
            #Calculate BRS
            self.brs = self.compute_brs(sbp,rr)

    def compute_sbp(self, art_signal):
        peaks, _ = find_peaks(art_signal, distance=100)
        sbp = art_signal[peaks]
        return sbp

    def compute_brs(self, sbp, rr):
        brs = []
        n = min(len(sbp), len(rr)) - 2
        for i in range(n):
            if sbp[i] < sbp[i+1] < sbp[i+2] and rr[i] < rr[i+1] < rr[i+2]:
                slope, _, r_value, _, _ = linregress(sbp[i:i+3], rr[i:i+3])
                if r_value > 0.6:
                    brs.append(slope)
            elif sbp[i] > sbp[i+1] > sbp[i+2] and rr[i] > rr[i+1] > rr[i+2]:
                slope, _, r_value, _, _ = linregress(sbp[i:i+3], rr[i:i+3])
                if r_value > 0.6:
                    brs.append(slope)

        if len(brs) > 0:
            print("BRS promedio:", np.mean(brs))
            print("BRS:", [float(x) for x in brs])
        else:
            print("No se encontraron secuencias válidas para BRS.")
        return np.array(brs)
