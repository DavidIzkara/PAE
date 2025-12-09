import vitaldb
import pandas as pd

class DynamicCompliance:

    def __init__(self, data):

        if isinstance(data, vitaldb.VitalFile):
            self._from_vf(data)
        else:
            self._from_df(data)

    def _from_vf(self, vf):
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
        self.values = pd.DataFrame({'Timestamp': pre_dc["Time"], 'DC': (pre_dc[tv_track] / (pre_dc[pip_track] - pre_dc[peep_track]))})


    def _from_df(self, list_dataframe: dict[str, pd.DataFrame]):
        # Get a Dataframes dictionary

        tv = list_dataframe['Intellivue/TV_EXP']
        pip = list_dataframe['Intellivue/PIP_CMH2O']
        peep = list_dataframe['Intellivue/PEEP_CMH2O']

        # Deletes the nan values
        tv_clean = tv[tv["value"].notna()]
        pip_clean = pip[pip["value"].notna()]
        peep_clean = peep[peep["value"].notna()]

        # Creates a new dataframe with timestamp | tv_value | pip_value | peep_value where the 3 values come from the same timestamp
        pre_dc = tv_clean.merge(pip_clean, on="time_ms").merge(peep_clean, on="time_ms")

        #Creates the DP dataframe: Timestamp | DC_value
        self.values = pd.DataFrame({'Timestamp': pre_dc["time_ms"], 'DC': pre_dc["value_x"] / (pre_dc["value_y"] - pre_dc["value"])})

#Does not require any special handling of missing data as this class is only used when we have the data.
#Does not need to handle multiple possible track names as these are fixed.
