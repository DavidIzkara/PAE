import numpy as np
from scipy.stats import linregress
import vitaldb
from scipy.signal import find_peaks
from compute_rr import compute_rr
from utils_zarr_corrected import leer_senyal,list_available_tracks

def vital_brs(vital_file_path):
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
    
    art_track = next(
            (t for t in available_tracks if 'Intellivue/ART' in t),
                next((t for t in available_tracks if 'Intellivue/ABP' in t), None))
    if art_track is None:
        print("No se encontraron señales ART.")
        return None

    ecg=vital_file.to_pandas(track_name=ecg_track,interval=1/500)
    art=vital_file.to_pandas(track_name=art_track,interval=1/500)
    rr= compute_rr(ecg)
    sbp= compute_sbp(art)
    brs= compute_brs(sbp,rr)
    return brs

def zarr_brs(zarr_path):
    track,_ = list_available_tracks(zarr_path)
    print(track)
    ecg_candidates = ['Intellivue/ECG_I', 'Intellivue/ECG_II', 'Intellivue/ECG_III', 'Intellivue/ECG_V']
    ecg_track = next((t for cand in ecg_candidates for t in track if cand in t),None)

    if ecg_track is None:
        print("No se encontraron señales ECG.")
        return None
    art_track = next(
        (t for t in track if 'Intellivue/ART' in t),
            next((t for t in track if 'Intellivue/ABP' in t), None))
    if art_track is None:
        print("No se encontraron señales ART.")
        return None
    
    ecg= leer_senyal(zarr_path,ecg_track)
    art= leer_senyal(zarr_path,art_track)
    rr= compute_rr(ecg)
    sbp= compute_sbp(art)
    brs= compute_brs(sbp,rr)
    return brs

def compute_brs(sbp,rr):
    #Sequence method for BRS calculation
    brs = []
    n = min(len(sbp), len(rr)) - 2
    for i in range(n):
        # Ascending sequence
        if sbp[i] < sbp[i+1] < sbp[i+2] and rr[i] < rr[i+1] < rr[i+2]:
            slope,intercept,r_value,p,std_err = linregress(sbp[i:i+3], rr[i:i+3])
            if r_value > 0.6:
                brs.append(slope)
        # Descending sequence
        elif sbp[i] > sbp[i+1] > sbp[i+2] and rr[i] > rr[i+1] > rr[i+2]:
            slope,intercept,r_value,p,std_err = linregress(sbp[i:i+3], rr[i:i+3])
            if r_value > 0.6:
                brs.append(slope)
        
    if len(brs) > 0:
        print("BRS promedio:", np.mean(brs))
        print("BRS:", [float(x) for x in brs])
    else:
        print("No se encontraron secuencias válidas para BRS.")

    return np.array(brs)

def compute_sbp(art_signal):
    #Find systolic peaks in the ART signal
    peaks,_=find_peaks(art_signal,distance=100)
    sbp=art_signal[peaks]
    return sbp


if __name__ == "__main__":
    #Example 
    zarr_path = "C:/Users/junle/Desktop/PAE/VitalParser_ART-Prediction_Algorithms/Algorithms/BOX2.zarr"
    import os

    zarr_path = "C:/Users/junle/Desktop/PAE/VitalParser_ART-Prediction_Algorithms/Algorithms/BOX2.zarr"
    print("Contenido de la carpeta principal:", os.listdir(zarr_path))
    signals_path = os.path.join(zarr_path, "signals")
    if os.path.exists(signals_path):
        print("Contenido de signals:", os.listdir(signals_path))
    else:
        print("No existe la carpeta 'signals'.")

    zarr_brs(zarr_path)
