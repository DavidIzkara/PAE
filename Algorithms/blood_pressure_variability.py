from vitaldb import VitalFile
import numpy as np
import pandas as pd

from compute_rr import compute_rr

SAMPLING_RATE = 500  # Hz
WINDOW_SIZE_RR = 5  # Number of RR intervals in each window

#* Window size set to 5 intervals, which averages to 5 seconds for a heart rate of 60 bpm and works for both rr and abp metrics.

class BloodPressureVariability:

    def __init__(self, data):

        if isinstance(data, VitalFile):
            self._from_vf(data)
        else:
            self._from_df(data)

    def _from_vf(self, vf: VitalFile):

        available_tracks = vf.get_track_names()

        hr_track = next(
            (t for t in available_tracks if 'Intellivue/ECG_II' in t),
            next((t for t in available_tracks if 'Intellivue/ECG_III' in t),
                 next((t for t in available_tracks if 'Intellivue/ECG_V' in t),
                      next((t for t in available_tracks if 'Intellivue/ECG_I' in t), None))))

        #Compute RR intervals
        hr = vf.to_pandas(track_names=hr_track, interval=1/SAMPLING_RATE, return_timestamp=True)
        rr = compute_rr(hr, hr_track)

        #Construct ABP dataframe and drop NaN values
        abp_raw = vf.to_numpy('Intellivue/ABP', interval=1, return_timestamp=True)
        abp = pd.DataFrame({ 'Intellivue/ABP': abp_raw[:,1], 'Time': abp_raw[:,0] })
        abp = abp.dropna()

        # Calculate HRV metrics
        metrics = self.compute_metrics(rr, abp, window=WINDOW_SIZE_RR)
        self.values = pd.DataFrame({'Time_ini_ms': metrics["Time_ini_ms"],
                                    'Time_fin_ms': metrics["Time_fin_ms"],
                                    'STD': metrics["std"],
                                    'CV': metrics["cv"],
                                    'ARV': metrics["arv"],
                                    'SDNN': metrics["sdnn"],
                                    'RMSSD': metrics["rmssd"]})

        return self.values

    def _from_df(self, list_dataframe: list[pd.DataFrame]):

        available_tracks = list_dataframe.keys() # type: ignore
        hr_track = next(
            (t for t in available_tracks if 'Intellivue/ECG_I' in t),
            next((t for t in available_tracks if 'Intellivue/ECG_II' in t),
                 next((t for t in available_tracks if 'Intellivue/ECG_III' in t),
                      next((t for t in available_tracks if 'Intellivue/ECG_V' in t), None))))

        # Compute RR intervals
        hr_raw = list_dataframe[hr_track] # type: ignore
        hr = pd.DataFrame({hr_track:hr_raw["value"], 'Time': hr_raw["time_ms"] })
        rr = compute_rr(hr, hr_track)

        # Construct ABP dataframe and drop NaN values
        abp_track = next((t for t in available_tracks if 'Intellivue/ABP' in t), None)
        abp_raw = list_dataframe[abp_track]  # type: ignore
        abp = pd.DataFrame({abp_track:abp_raw["value"], 'Time': abp_raw["time_ms"]})
        abp = abp.rename(columns={abp_track: 'Intellivue/ABP'}).dropna()

        # Calculate HRV metrics
        metrics = self.compute_metrics(rr, abp, window=WINDOW_SIZE_RR)
        self.values = pd.DataFrame({'Time_ini_ms': metrics["Time_ini_ms"],
                                    'Time_fin_ms': metrics["Time_fin_ms"],
                                    'STD': metrics["std"],
                                    'CV': metrics["cv"],
                                    'ARV': metrics["arv"],
                                    'SDNN': metrics["sdnn"],
                                    'RMSSD': metrics["rmssd"]})

        return self.values

    def compute_metrics(self, rr_df: pd.DataFrame, abp_df: pd.DataFrame, window=WINDOW_SIZE_RR):

        # Extract ABP values and times
        abp_vals = abp_df["Intellivue/ABP"].values
        abp_time = abp_df["Time"].values

        # Extract RR intervals and times
        rr = rr_df['rr'].values
        ts_ini = rr_df["Time_ini_ms"].values
        ts_fin = rr_df["Time_fin_ms"].values

        n = len(rr)
        results = []

        for i in range(n - window + 1):

            #Compute RR metrics using rolling window
            w = rr[i:i+window]
            sdnn = np.std(w, ddof=1) #type: ignore
            rmssd = np.sqrt(np.mean(np.square(np.diff(w))))  # type: ignore

            # Get window start and end times
            win_ini = ts_ini[i]
            win_fin = ts_fin[i + window - 1]

            # Get ABP values within the RR window
            mask_abp = (abp_time >= win_ini) & (abp_time <= win_fin)
            w = abp_vals[mask_abp]

            # Compute ABP metrics if enough data points are available
            if len(w) < 2:
                std = np.nan
                cv = np.nan
                arv = np.nan
            else:
                std = np.std(w, ddof=1)
                cv = std / np.mean(w)
                arv = np.mean(np.abs(np.diff(w)))

            results.append([win_ini, win_fin, std, cv, arv, sdnn, rmssd])

        return pd.DataFrame(results, columns=["Time_ini_ms", "Time_fin_ms", "std", "cv", "arv", "sdnn", "rmssd"])