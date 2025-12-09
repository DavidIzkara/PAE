import numpy as np
import pandas as pd
from utils_AL import Detectors

def compute_rr(signal, track):

    # print("Signal", signal)
    signal_clean = signal[signal[track].notna()]
    # print("Limpiado", signal_clean)

    ecg_signal = np.array(signal_clean[track], dtype=np.float64)
    timestamps = np.array(signal_clean["Time"], dtype=np.float64)
    # print("ecg_signal: ", ecg_signal)
    # print("timestamp_indexes: ", timestamps)

    # Generar el vector de tiempos
    times = np.arange(len(ecg_signal)) / 500

    # Detector Pan-Tompkins
    detectors = Detectors(500)
    r_peaks_ind = detectors.pan_tompkins_detector(ecg_signal)

    #A raíz de los indíces seleccionar el timestamp
    timestamps_indexes = timestamps[r_peaks_ind]
    # Calcula los intervalos R-R (segundos)
    r_peaks_times = times[r_peaks_ind]
    
    # print("lenght de timestamps_indexes: ", len(timestamps_indexes))
    # print("lenght de rr:", len(np.diff(r_peaks_times)))

    #Df tq: Timestamp_ini | Timestamp_fin | rr

    return pd.DataFrame({'Time_ini_ms': np.delete(timestamps_indexes, len(timestamps_indexes)-1 ), 'Time_fin_ms': np.delete(timestamps_indexes, 0), 'rr': np.diff(r_peaks_times)})
