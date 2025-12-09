import vitaldb
import pandas as pd
from cardiac_output import CardiacOutput

class SystemicVascularResistance:
    def __init__(self, data):

        if isinstance(data, vitaldb.VitalFile):
            self._from_vf(data)
        else:
            self._from_df(data)

    def _from_vf(self, vf):
        # Get all available track names in the VitalFile
        available_tracks = vf.get_track_names()

        # Try to find mean pressure tracks
        mean_track = next(
            (t for t in available_tracks if 'Intellivue/ABP_MEAN' in t), # First try for invasive mean BP
            next((t for t in available_tracks if 'Intellivue/BP_MEAN' in t), # Then try for another possible invasive mean BP
                next((t for t in available_tracks if 'Intellivue/NIBP_MEAN' in t), None))) # Finally try for non-invasive mean BP
        

        # Converts the signals to pandas dataframes
        mean = vf.to_pandas(track_names=mean_track, interval=0, return_timestamp=True)
        cvp_track = 'Intellivue/CVP_MEAN'
        cvp = vf.to_pandas(track_names=cvp_track, interval=0, return_timestamp=True)

        
        # Deletes the nan values
        mean_clean = mean[mean[mean_track].notna()]
        cvp_clean = cvp[cvp[cvp_track].notna()]
        
        co = CardiacOutput(vf).values
        # Creates a new dataframe with timestamp | mean_value | cvp_value | co_value where the 3 values come from the same timestamp
        pre_svr= mean_clean.merge(cvp_clean, on="Time").merge(co, left_on="Time", right_on = 'Timestamp')

        #Creates the SVR dataframe: Timestamp | SVR_value
        self.values = pd.DataFrame({'Timestamp': pre_svr["Time"], 'SVR': ((pre_svr[mean_track] - pre_svr[cvp_track])*80)/pre_svr['CO']})


    def _from_df(self, list_dataframe: dict[pd.DataFrame]):
        # Get a Dataframes dictionary
        # Get the track names
        available_tracks = list_dataframe.keys()

        # Try to find mean pressure tracks
        mean_track = next(
            (t for t in available_tracks if 'Intellivue/ABP_MEAN' in t), # First try for invasive mean BP
            next((t for t in available_tracks if 'Intellivue/BP_MEAN' in t), # Then try for another possible invasive mean BP
                next((t for t in available_tracks if 'Intellivue/NIBP_MEAN' in t), None))) # Finally try for non-invasive mean BP
        

        # Converts the signals to pandas dataframes
        mean = list_dataframe[mean_track] 
        cvp_track = 'Intellivue/CVP_MEAN'
        cvp = list_dataframe[cvp_track] 
        
        # Deletes the nan values
        mean_clean = mean[mean["value"].notna()]
        cvp_clean = cvp[cvp["value"].notna()]
        
        co = CardiacOutput(list_dataframe).values
        # Creates a new dataframe with timestamp | mean_value | cvp_value | co_value where the 3 values come from the same timestamp
        pre_svr= mean_clean.merge(cvp_clean, on="time_ms").merge(co, left_on="time_ms", right_on = 'Timestamp')

        #Creates the SVR dataframe: Timestamp | SVR_value
        self.values = pd.DataFrame({'Timestamp': pre_svr["time_ms"], 'SVR': ((pre_svr["value_x"] - pre_svr["value_y"])*80)/pre_svr['CO']}) 

#Calculates Systemic Vascular Resistance using Mean Arterial Pressure, Central Venous Pressure, and Cardiac Output.
#Handles multiple possible mean arterial pressure track names for robustness.
