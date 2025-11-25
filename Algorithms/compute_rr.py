import numpy as np
from ecgdetectors import Detectors

def compute_rr(signal):

    ecg_signal = np.array(signal, dtype=np.float64)
    ecg_signal_clean = ecg_signal[~np.isnan(ecg_signal)]

    # Generar el vector de tiempos
    times = np.arange(len(ecg_signal_clean)) / 500

    # Detector Pan-Tompkins
    detectors = Detectors(500)
    r_peaks_ind = detectors.pan_tompkins_detector(ecg_signal_clean)

    # Calcula los intervalos R-R (segundos)
    r_peaks_times = times[r_peaks_ind]

    return np.diff(r_peaks_times)
