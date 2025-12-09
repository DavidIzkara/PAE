import vitaldb
import pandas as pd

class EffectiveArterialElastance:
    def __init__(self, data):

        if isinstance(data, vitaldb.VitalFile):
            self._from_vf(data)
        else:
            self._from_df(data)


    def _from_vf(self, vf):
        # Get all available track names in the VitalFile
        available_tracks = vf.get_track_names()

        # Try to find systolic pressure tracks
        sys_track = next(
            (t for t in available_tracks if 'Intellivue/ABP_SYS' in t), # First try for invasive systolic BP
            next((t for t in available_tracks if 'Intellivue/BP_SYS' in t), # Then try for another possible invasive systolic BP
                next((t for t in available_tracks if 'Intellivue/NIBP_SYS' in t), None))) # Finally try for non-invasive systolic BP
        

        bld_track = 'Intellivue/VOL_BLD_STROKE'

        # Converts the signals to pandas dataframes
        sys = vf.to_pandas(track_names=sys_track, interval=0, return_timestamp=True)
        bld = vf.to_pandas(track_names=bld_track, interval=0, return_timestamp=True)

        
        # Deletes the nan values
        sys_clean = sys[sys[sys_track].notna()]
        bld_clean = bld[bld[bld_track].notna()]

        # Creates a new dataframe with timestamp | sys_value | bld_value where both values come from the same timestamp
        pre_eae= sys_clean.merge(bld_clean, on="Time")

        #Creates the EAE dataframe: Timestamp | EAE_value
        self.values = pd.DataFrame({'Timestamp': pre_eae["Time"], 'EAE': (0.9 * pre_eae[sys_track]) / pre_eae[bld_track]})


    def _from_df(self, list_dataframe: dict[pd.DataFrame]):
        # Get a Dataframes dictionary
        # Get the track names
        available_tracks = list_dataframe.keys()

         # Try to find systolic pressure tracks
        sys_track = next(
            (t for t in available_tracks if 'Intellivue/ABP_SYS' in t), # First try for invasive systolic BP
            next((t for t in available_tracks if 'Intellivue/BP_SYS' in t), # Then try for another possible invasive systolic BP
                next((t for t in available_tracks if 'Intellivue/NIBP_SYS' in t), None))) # Finally try for non-invasive systolic BP
        

        bld_track = 'Intellivue/VOL_BLD_STROKE'

        # Converts the signals to pandas dataframes
        sys = list_dataframe[sys_track]
        bld = list_dataframe[bld_track]

        
        # Deletes the nan values
        sys_clean = sys[sys["value"].notna()]
        bld_clean = bld[bld["value"].notna()]

        # Creates a new dataframe with timestamp | sys_value | bld_value where both values come from the same timestamp
        pre_eae = sys_clean.merge(bld_clean, on="time_ms")

        #Creates the EAE dataframe: Timestamp | EAE_value
        self.values = pd.DataFrame({'Timestamp': pre_eae["time_ms"], 'EAE': (0.9 * pre_eae["value_x"]) / pre_eae["value_y"]})
