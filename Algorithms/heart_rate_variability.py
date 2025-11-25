import vitaldb
import numpy as np
import pandas as pd
from compute_rr import compute_rr


class HeartRateVariability:

    def __init__(self, data):

        if isinstance(data, vitaldb.VitalFile):
            self._from_vf(data)
        else:
            self._from_df(data)

    def _from_vf(self, vf):
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
        self.sdnn = self.compute_sdnn(rr)
        self.rmssd = self.compute_rmssd(rr)
        self.pnn50 = self.compute_pnn50(rr)
        #Como no está en el values es un objeto de la clase que no se podrá printear desde fuera xq 
        #no es un pandas, en caso de hacerlo igual que el resto, con un self.values = pd.DataFrame,
        #Funcionará igual, como no se podrá mostrar por pantalla, se dejará inhabilitado.

    def _from_df(self, list_dataframe: list[pd.DataFrame]):
        #Se recibe una lista de dataframes
        #Se sacan los indices, que son los nombres de las variables
        available_tracks = list_dataframe.keys()

        # Try to find heart rate wave
        hr_track = next(
            (t for t in available_tracks if 'Intellivue/ECG_I' in t), 
            next((t for t in available_tracks if 'Intellivue/ECG_II' in t),     
                 next((t for t in available_tracks if 'Intellivue/ECG_III' in t), 
                      next((t for t in available_tracks if 'Intellivue/ECG_V' in t), None)))) 
        
        hr = list_dataframe[hr_track]
        rr = compute_rr(hr)

        # Calculate HRV metrics
        self.sdnn = self.compute_sdnn(rr)
        self.rmssd = self.compute_rmssd(rr)
        self.pnn50 = self.compute_pnn50(rr)
        #Como no está en el values es un objeto de la clase que no se podrá printear desde fuera xq 
        #no es un pandas, en caso de hacerlo igual que el resto, con un self.values = pd.DataFrame,
        #Funcionará igual, como no se podrá mostrar por pantalla, se dejará inhabilitado.

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

    
