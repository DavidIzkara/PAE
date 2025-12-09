import vitaldb
import pandas as pd

class CardiacOutput:
    def __init__(self, data):
        if isinstance(data, vitaldb.VitalFile):
            self._from_vf(data)
        else:
            self._from_df(data)


    def _from_vf(self, vf):
         # Get all available track names in the VitalFile
        available_tracks = vf.get_track_names()

        # Try to find heart rate tracks
        hr_track = next(
            (t for t in available_tracks if 'Intellivue/ECG_HR' in t),          # First try for ECG_HR
            next((t for t in available_tracks if 'Intellivue/ABP_HR' in t),     # Then try for ABP_HR
                 next((t for t in available_tracks if 'Intellivue/HR' in t), None))) # Finally try for generic HR track

        # Converts the signals to pandas dataframes
        hr = vf.to_pandas(track_names=hr_track, interval=0, return_timestamp=True)
        bld_track = 'Intellivue/VOL_BLD_STROKE'
        bld = vf.to_pandas(track_names=bld_track, interval=0, return_timestamp=True)


        # Deletes the nan values
        hr_clean = hr[hr[hr_track].notna()]
        bld_clean = bld[bld[bld_track].notna()]

        # Creates a new dataframe with timestamp | bld_value | bld_value where both values come from the same timestamp
        pre_co= bld_clean.merge(hr_clean, on="Time")

        # Creates the CO dataframe: Timestamp | CO_value
        self.values = pd.DataFrame({'Timestamp': pre_co["Time"], 'CO': pre_co[bld_track] * pre_co[hr_track]})

    def _from_df(self, list_dataframe: dict[str, pd.DataFrame]):
        # Get a Dataframes dictionary
        # Get the track names
        available_tracks = list_dataframe.keys()

        # Try to find heart rate tracks
        hr_track = next(
            (t for t in available_tracks if 'Intellivue/ECG_HR' in t),          # First try for ECG_HR
            next((t for t in available_tracks if 'Intellivue/ABP_HR' in t),     # Then try for ABP_HR
                 next((t for t in available_tracks if 'Intellivue/HR' in t), None))) # Finally try for generic HR track

        bld_track = 'Intellivue/VOL_BLD_STROKE'

        assert hr_track is not None
        hr = list_dataframe[hr_track]
        bld = list_dataframe[bld_track]

        # Deletes the nan values
        hr_clean = hr[hr["value"].notna()]
        bld_clean = bld[bld["value"].notna()]

        # Creates a new dataframe with timestamp | bld_value | bld_value where both values come from the same timestamp
        pre_co= bld_clean.merge(hr_clean, on="time_ms")

        #Creates the CO dataframe: Timestamp | CO_value
        self.values = pd.DataFrame({'Timestamp': pre_co["time_ms"], 'CO': pre_co["value_x"] * pre_co["value_y"]})

#Calculates Cardiac Output by multiplying Stroke Volume by Heart Rate.
#Handles multiple possible heart rate track names for robustness.
