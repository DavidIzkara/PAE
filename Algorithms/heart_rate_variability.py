import vitaldb
import numpy as np
from compute_rr import compute_rr


class HeartRateVariability:

    def __init__(self, vf: vitaldb.VitalFile):
        # Get all available track names in the VitalFile
        available_tracks = vf.get_track_names()

        # Try to find heart rate wave
        hr_track = next(
            (t for t in available_tracks if 'Intellivue/ECG_I' in t), 
            next((t for t in available_tracks if 'Intellivue/ECG_II' in t),     
                 next((t for t in available_tracks if 'Intellivue/ECG_III' in t), 
                      next((t for t in available_tracks if 'Intellivue/ECG_V' in t), None)))) 

        # Convert the signals to NumPy arrays
        hr = vf.to_pandas(track_names=hr_track, interval=1/500)
        rr = compute_rr(hr)

        # Calculate HRV metrics
        sdnn = self.compute_sdnn(rr)
        rmssd = self.compute_rmssd(rr)
        pnn50 = self.compute_pnn50(rr)

    def compute_sdnn(self, rr, window=5):
        n = len(rr)
        result = []
        for i in range(n - window + 1):
            w = rr[i : i + window]
            result.append(np.std(w, ddof=1))
        return np.array(result)

    def compute_rmssd(self, rr, window=5):
        n = len(rr)
        result = []
        for i in range(n - window + 1):
            w = rr[i : i + window]
            diffs = np.diff(w)
            result.append(np.sqrt(np.mean(diffs ** 2)))
        return np.array(result)

    def compute_pnn50(self, rr, window=5, threshold=50):
        n = len(rr)
        result = []
        for i in range(n - window + 1):
            w = rr[i : i + window]
            diffs = np.abs(np.diff(w))
            count = np.sum(diffs > threshold)
            result.append(count / (len(diffs)) * 100)
        return np.array(result)

    
