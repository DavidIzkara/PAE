import numpy as np
from ecgdetectors import Detectors

def compute_rr(signal):

    print("Signal", signal)
    ecg_signal = np.array(signal, dtype=np.float64)
    print("ecg_signal", ecg_signal)
    ecg_signal_clean = ecg_signal[~np.isnan(ecg_signal)]

    # Generar el vector de tiempos
    times = np.arange(len(ecg_signal_clean)) / 500

    # Detector Pan-Tompkins
    detectors = Detectors(500)
    r_peaks_ind = detectors.pan_tompkins_detector(ecg_signal_clean)

    #A raíz de los indíces seleccionar el timestamp

    # Calcula los intervalos R-R (segundos)
    r_peaks_times = times[r_peaks_ind]

    #una vez se ha calculado el intervalo hacer un df tq: Timestamp_ini | Timestamp_fin | rr

    return np.diff(r_peaks_times)
