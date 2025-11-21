import vitaldb

class DynamicCompliance: 
        
    def __init__(self, vf: vitaldb.VitalFile):
    
        tv_track='Intellivue/TV_EXP'
        pip_track='Intellivue/PIP_CMH2O'
        peep_track='Intellivue/PEEP_CMH2O'

        # Converts the signals to pandas dataframes
        tv = vf.to_pandas(track_names=tv_track, interval=0, return_timestamp=True)
        pip = vf.to_pandas(track_names=pip_track, interval=0, return_timestamp=True)
        peep = vf.to_pandas(track_names=peep_track, interval=0, return_timestamp=True)
  
        # Deletes the nan values
        tv_clean = tv[tv[tv_track].notna()]
        pip_clean = pip[pip[pip_track].notna()]
        peep_clean = peep[peep[peep_track].notna()]
        
        # Creates a new dataframe with timestamp | tv_value | pip_value | peep_value where the 3 values come from the same timestamp
        pre_dc= tv_clean.merge(pip_clean, on="Time").merge(peep_clean, on="Time")

        #Creates the DC dataframe: Timestamp | DC_value
        self.values = {'Timestamp': pre_dc["Time"], 'DC': (pre_dc[tv_track] / (pre_dc[pip_track] - pre_dc[peep_track]))}

#Does not require any special handling of missing data as this class is only used when we have the data.
#Does not need to handle multiple possible track names as these are fixed.
