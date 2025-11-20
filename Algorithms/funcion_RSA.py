import numpy as np
from scipy.signal import find_peaks
import vitaldb
from compute_rr import compute_rr 
from utils_zarr_corrected import list_available_tracks,leer_senyal

def vital_rsa(vital_file_path):
    vital_file = vitaldb.VitalFile(vital_file_path)
    available_tracks= vital_file.get_track_names()

    ecg_track = next(
    (t for t in available_tracks if 'Intellivue/ECG_I' in t),
    next((t for t in available_tracks if 'Intellivue/ECG_II' in t),
        next((t for t in available_tracks if 'Intellivue/ECG_III' in t),
            next((t for t in available_tracks if 'Intellivue/ECG_V' in t), None))))
    if ecg_track is None:
        print("No se encontraron señales ECG.")
        return None
    
    resp_track = next(
            (t for t in available_tracks if 'Intellivue/RESP' in t),
                next((t for t in available_tracks if 'Intellivue/CO2' in t), None))
    if resp_track is None:
        print("No se encontraron señales RESP.")
        return None
    
    ecg=vital_file.to_pandas(track_name=ecg_track,interval=1/500)
    resp=vital_file.to_pandas(track_name=resp_track,interval=1/500)
    rr= compute_rr(ecg)
    rsa= compute_rsa(rr,resp)

    return rsa

def zarr_rsa(zarr_path):
    track,_ = list_available_tracks(zarr_path)
    ecg_track = next((t for t in track if 'Intellivue/ECG_I' in t),
                    next((t for t in track if 'Intellivue/ECG_II' in t),
                        next((t for t in track if 'Intellivue/ECG_III' in t),
                            next((t for t in track if 'Intellivue/ECG_V' in t), None))))
    if ecg_track is None:
        print("No se encontraron señales ECG.")
        return None
    
    resp_track = next((t for t in track if 'Intellivue/RESP' in t),
                    next((t for t in track if 'Intellivue/CO2' in t), None))
    if resp_track is None:
        print("No se encontraron señales RESP.")
        return None
    
    ecg= leer_senyal(zarr_path,ecg_track)
    resp= leer_senyal(zarr_path,resp_track)
    rr= compute_rr(ecg)
    rsa= compute_rsa(rr,resp)
    return rsa

def compute_rsa(rr_intervals,resp_signal):
    
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

if __name__== "__main__":
    vital_file = "C:/Users/junle/Desktop/PAE/VitalParser_ART-Prediction_Algorithms/PAE_NEW/VitalDB_data/230718/QUI12_230718_000947.vital"
    rsa_result = vital_rsa(vital_file)
    print("RSA values (seconds):", rsa_result)
