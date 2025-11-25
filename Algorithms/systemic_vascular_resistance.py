import vitaldb
from cardiac_output import CardiacOutput

class SystemicVascularResistance:
    def __init__(self, vf: vitaldb.VitalFile):
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
        # Creates a new dataframe with timestamp | mean_value | cvp_value | co_value where both values come from the same timestamp
        pre_svr= mean_clean.merge(cvp_clean, on="Time").merge(co, left_on="Time", right_on = 'Timestamp')

        #Creates the SVR dataframe: Timestamp | SVR_value
        self.values = {'Timestamp': pre_svr["Time"], 'SVR': ((pre_svr[mean_track] - pre_svr[cvp_track])*80)/pre_svr['CO']} 

#Calculates Systemic Vascular Resistance using Mean Arterial Pressure, Central Venous Pressure, and Cardiac Output.
#Handles multiple possible mean arterial pressure track names for robustness.
