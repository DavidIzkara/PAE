from ecgdetectors import Detectors
import vitaldb
import numpy as np

SAMPLING_RATE = 500  # Hz

def funcion_rr(hr):

    ecg_signal = np.array(hr, dtype=np.float64)
    ecg_signal = ecg_signal[~np.isnan(ecg_signal)]

    # Generar el vector de tiempos
    times = np.arange(len(ecg_signal)) / SAMPLING_RATE

    # Detector Pan-Tompkins
    
    detectors = Detectors(SAMPLING_RATE) # type: ignore
    r_peaks_ind = detectors.pan_tompkins_detector(ecg_signal)

    # Calcula los intervalos R-R (segundos)
    r_peaks_times = times[r_peaks_ind]
    return np.diff(r_peaks_times)

#Standard Deviation
def std(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return variance ** 0.5 

#Coefficient of Variation
def cv(values):
    mean = sum(values) / len(values)
    return std(values) / mean if mean != 0 else float('nan')

#Average Real Variability
def arv(values):
    arv_sum = sum(abs(values[i] - values[i-1]) for i in range(1, len(values)))
    return arv_sum / (len(values) - 1) if len(values) > 1 else float('nan')

#Standard Deviation of RR intervals
def sdnn(values):
    return std(values)

#Root Mean Square of Successive Differences
def rmssd(values):

    n = len(values)
    if n < 2:
        return float('nan')
    squared_diffs = [(values[i] - values[i-1]) ** 2 for i in range(1, n)]
    mean_squared_diff = sum(squared_diffs) / (n - 1)
    return mean_squared_diff ** 0.5

class BloodPressureVariability:
    def __init__(self, vital_path):
            
        vf = vitaldb.VitalFile(vital_path)
        available_tracks = vf.get_track_names()

        # Try to find heart rate wave
        hr_track = next(
            (t for t in available_tracks if 'Intellivue/ECG_I' in t), 
            next((t for t in available_tracks if 'Intellivue/ECG_II' in t),     
                    next((t for t in available_tracks if 'Intellivue/ECG_III' in t), None))) 

        hr = vf.to_pandas(track_names=hr_track, interval=1/SAMPLING_RATE)
        abp_raw = vf.to_numpy('Intellivue/ABP', interval=1.0)
        self.abp_values = np.asarray(abp_raw, dtype=float)
        
        rr = funcion_rr(hr)

        # compute scalar metrics (cast to Python floats so savetxt gets homogeneous shape)
        metrics_list = [
            float(std(self.abp_values)) if self.abp_values.size > 0 else float('nan'),
            float(cv(self.abp_values)) if self.abp_values.size > 0 else float('nan'),
            float(arv(self.abp_values)) if self.abp_values.size > 1 else float('nan'),
            float(sdnn(rr)) if rr.size > 0 else float('nan'),
            float(rmssd(rr)) if rr.size > 1 else float('nan'),
        ]

        self.values = metrics_list

        # Save as a single-row CSV (5 columns) with 3 significant figures
        np.savetxt('systolic_values.csv', np.asarray(self.values, dtype=float).reshape(1, -1), delimiter=',', fmt='%.3g')
        
if __name__ == "__main__":
    vital_path = "n5j8vrrsb_250630_130506.vital"
    bpv = BloodPressureVariability(vital_path)
    # Print metrics with 3 significant figures
    formatted = ', '.join(f"{v:.3g}" for v in bpv.values)
    print("metrics (std, cv, arv, sdnn, rmssd):", formatted)