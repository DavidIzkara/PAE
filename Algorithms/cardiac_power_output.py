import vitaldb
from cardiac_output import CardiacOutput

class CardiacPowerOutput:
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
        
        # Deletes the nan values
        mean_clean = mean[mean[mean_track].notna()]
        
        co = CardiacOutput(vf).values
        # Creates a new dataframe with timestamp | mean_value | CO_value where both values come from the same timestamp
        pre_cpo= mean_clean.merge(co, left_on="Time", right_on = 'Timestamp')

        #Creates the CPO dataframe: Timestamp | CPO_value
        self.values = {'Timestamp': pre_cpo["Time"], 'CPO': (pre_cpo[mean_track] * pre_cpo['CO'])/ 451.0} 

#Calculates Cardiac Power Output by multiplying Mean Arterial Pressure by Cardiac Output and dividing by 451.
#Handles multiple possible mean pressure track names for robustness.
