import vitaldb

class DrivingPressure: 
        
    def __init__(self, vf: vitaldb.VitalFile):

        pplat_track='Intellivue/PPLAT_CMH2O'
        peep_track='Intellivue/PEEP_CMH2O'
        # Converts the signals to pandas dataframes
        pplat = vf.to_pandas(track_names=pplat_track, interval=0, return_timestamp=True)
        peep = vf.to_pandas(track_names=peep_track, interval=0, return_timestamp=True)

        
        # Deletes the nan values
        pplat_clean = pplat[pplat[pplat_track].notna()]
        peep_clean = peep[peep[peep_track].notna()]
        
        # Creates a new dataframe with timestamp | pplat_value | peep_value where both values come from the same timestamp
        pre_dp= pplat_clean.merge(peep_clean, on="Time")

        #Creates the DP dataframe: Timestamp | DP_value
        self.values = {'Timestamp': pre_dp["Time"], 'DP': pre_dp[pplat_track] - pre_dp[peep_track]} 
#Does the driving pressure calculation by subtracting PEEP from PLAT.
#Does not require any special handling of missing data as this class is only used when we have the data.
#Does not need to handle multiple possible track names as these are fixed.
