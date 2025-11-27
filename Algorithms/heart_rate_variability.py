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
        hr = vf.to_pandas(track_names=hr_track, interval=1/500, return_timestamp=True)
        rr = compute_rr(hr, hr_track)

        # Calculate HRV metrics
        sdnn = self.compute_sdnn(rr)
        rmssd = self.compute_rmssd(rr)
        pnn50 = self.compute_pnn50(rr)
        
        values = sdnn.merge(rmssd, on = "Time_ini_ms").merge(pnn50, on = "Time_ini_ms")
    
        self.values = pd.DataFrame({'Time_ini_ms': values["Time_ini_ms"], 'Time_fin_ms': values["Time_fin_ms"], 'SDNN': values["sdnn"], 'RMSSD': values["rmssd"], 'PNN50': values["pnn50"]})
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
        
        hr_raw = list_dataframe[hr_track]
        hr = pd.DataFrame({hr_track:hr_raw["value"], 'Time': hr_raw["time_ms"] })
        print("Zarr hr",hr)
        rr = compute_rr(hr, hr_track)

        # Calculate HRV metrics
        sdnn = self.compute_sdnn(rr)
        rmssd = self.compute_rmssd(rr)
        pnn50 = self.compute_pnn50(rr)
        
        values = sdnn.merge(rmssd, on = "Time_ini_ms").merge(pnn50, on = "Time_ini_ms")
        self.values = pd.DataFrame({'Time_ini_ms': values["Time_ini_ms"], 'Time_fin_ms': values["Time_fin_ms"], 'SDNN': values["sdnn"], 'RMSSD': values["rmssd"], 'PNN50': values["pnn50"]})
        #Como no está en el values es un objeto de la clase que no se podrá printear desde fuera xq 
        #no es un pandas, en caso de hacerlo igual que el resto, con un self.values = pd.DataFrame,
        #Funcionará igual, como no se podrá mostrar por pantalla, se dejará inhabilitado.

    def compute_sdnn(self, rr_df, window=5):
        rr = rr_df['rr'].values
        n = len(rr)

        # Extraer timestamps
        ts_ini = rr_df["Time_ini_ms"].values
        ts_fin = rr_df["Time_fin_ms"].values

        results = []

        for i in range(n - window + 1):
            w = rr[i : i + window]
            sdnn_value = np.std(w, ddof=1)

            # Primer timestamp_ini de la ventana
            win_ini = ts_ini[i]
            # Último timestamp_fin de la ventana
            win_fin = ts_fin[i + window - 1]

            results.append([win_ini, win_fin, sdnn_value])

        return pd.DataFrame(results, columns=["Time_ini_ms", "Time_fin_ms", "sdnn"])

    def compute_rmssd(self, rr_df, window=5):
        rr = rr_df['rr'].values
        ts_ini = rr_df["Time_ini_ms"].values
        ts_fin = rr_df["Time_fin_ms"].values

        n = len(rr)
        results = []

        for i in range(n - window + 1):
            w = rr[i : i + window]
            diffs = np.diff(w)
            rmssd_value = np.sqrt(np.mean(diffs ** 2))

            win_ini = ts_ini[i]                 # primer timestamp_ini de la ventana
            win_fin = ts_fin[i + window - 1]    # último timestamp_fin de la ventana

            results.append([win_ini, win_fin, rmssd_value])

        return pd.DataFrame(results, columns=["Time_ini_ms", "Time_fin_ms", "rmssd"])

    def compute_pnn50(self, rr_df, window=5, threshold=50):
        rr = rr_df['rr'].values
        ts_ini = rr_df["Time_ini_ms"].values
        ts_fin = rr_df["Time_fin_ms"].values

        n = len(rr)
        results = []

        for i in range(n - window + 1):
            w = rr[i : i + window]
            diffs = np.abs(np.diff(w))
            count = np.sum(diffs > threshold)
            pnn50_value = (count / len(diffs)) * 100

            win_ini = ts_ini[i]                 # primer timestamp_ini de la ventana
            win_fin = ts_fin[i + window - 1]    # último timestamp_fin de la ventana

            results.append([win_ini, win_fin, pnn50_value])

        return pd.DataFrame(results, columns=["Time_ini_ms", "Time_fin_ms", "pnn50"])


    
