import vitaldb
import numpy as np
import pandas as pd
from compute_rr import compute_rr


class HeartRateVariability:

    def __init__(self):
        self.last4_rr = []
        self.last4_ini = []
        self.values = None

    def compute(self, data):
        if isinstance(data, vitaldb.VitalFile):
            self._from_vf(data)
        else:
            self._from_df(data)
        return self.values

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
    
        self.values = self.compute_hrv(rr)


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
        rr = compute_rr(hr, hr_track)
        
        self.values = self.compute_hrv(rr)


    def compute_hrv(self, rr_df, window = 5, threshold=50):
        rr = rr_df['rr'].values
        n = len(rr)

        # Extraer timestamps
        ts_ini = rr_df["Time_ini_ms"].values
        ts_fin = rr_df["Time_fin_ms"].values

        results = []

        if len(self.last4_rr) == 0:

            n = len(rr)
            if n < window:
                return pd.DataFrame(columns=["Time_ini_ms", "Time_fin_ms", "sdnn", "rmsdd", "pnn50"])

            for i in range(n - window + 1):
                w = rr[i : i + window]
                #sdnn
                sdnn_value = np.std(w, ddof=1)

                #rmssd
                diffs = np.diff(w)
                rmssd_value = np.sqrt(np.mean(diffs ** 2))

                #pnn50
                count = np.sum(diffs > threshold)
                pnn50_value = (count / len(diffs)) * 100

                win_ini = ts_ini[i]
                win_fin = ts_fin[i + window - 1]

                results.append([win_ini, win_fin, sdnn_value, rmssd_value, pnn50_value])

            # Guardar últimos 4 valores para streaming
            self.last4_rr = rr[-(window-1):].tolist()
            self.last4_ini = ts_ini[-(window-1):].tolist()

            return pd.DataFrame(results, columns=["Time_ini_ms", "Time_fin_ms", "sdnn", "rmsdd", "pnn50"])

        else:
            for x, t_ini, t_fin in zip(rr, ts_ini, ts_fin):

                # Ventana = last4 + nuevo valor
                w = self.last4_rr + [x]
                #sdnn
                sdnn_value = np.std(w, ddof=1)

                #rmssd
                diffs = np.diff(w)
                rmssd_value = np.sqrt(np.mean(diffs ** 2))

                #pnn50
                count = np.sum(diffs > threshold)
                pnn50_value = (count / len(diffs)) * 100

                # Time_ini = el ini del valor más viejo guardado en last4
                win_ini = self.last4_ini[0]

                # Time_fin = el fin del nuevo valor recibido
                win_fin = t_fin

                results.append([win_ini, win_fin, sdnn_value, rmssd_value, pnn50_value])

                # Actualizar buffers last4 desplazando
                self.last4_rr.pop(0)
                self.last4_rr.append(x)

                self.last4_ini.pop(0)
                self.last4_ini.append(t_ini)

            return pd.DataFrame(results, columns=["Time_ini_ms", "Time_fin_ms", "sdnn", "rmsdd", "pnn50"])
