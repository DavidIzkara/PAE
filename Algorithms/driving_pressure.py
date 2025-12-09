import vitaldb
import pandas as pd

class DrivingPressure: 
        
    def __init__(self, data):

        if isinstance(data, vitaldb.VitalFile):
            self._from_vf(data)
        else:
            self._from_df(data)


    def _from_vf(self, vf):
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
        self.values = pd.DataFrame({'Timestamp': pre_dp["Time"], 'DP': pre_dp[pplat_track] - pre_dp[peep_track]})



    def _from_df(self, list_dataframe: dict[pd.DataFrame]):
        # Get a Dataframes dictionary
        
        pplat = list_dataframe['Intellivue/PPLAT_CMH2O'] 
        peep = list_dataframe['Intellivue/PEEP_CMH2O']

        # Deletes the nan values
        pplat_clean = pplat[pplat["value"].notna()]
        peep_clean = peep[peep["value"].notna()]

        # Creates a new dataframe with timestamp | pplat_value | peep_value where both values come from the same absolute timestamp
        pre_dp= pplat_clean.merge(peep_clean, on="time_ms")
        #print(pre_dp)

        #Creates the DP dataframe: Timestamp | DP_value
        self.values = pd.DataFrame({'Timestamp': pre_dp["time_ms"], 'DP': pre_dp["value_x"] - pre_dp["value_y"]}) 
#Does the driving pressure calculation by subtracting PEEP from PLAT.
#Does not require any special handling of missing data as this class is only used when we have the data.
#Does not need to handle multiple possible track names as these are fixed.
