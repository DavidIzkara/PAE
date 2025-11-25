import vitaldb

class RoxIndex:
    def __init__(self, vf: vitaldb.VitalFile):

        sato2_track='Intellivue/PLETH_SAT_O2'
        fio2_track='Intellivue/FiO2'
        # Converts the signals to pandas dataframes
        sato2 = vf.to_pandas(track_names=sato2_track, interval=0, return_timestamp=True)
        fio2 = vf.to_pandas(track_names=fio2_track, interval=0, return_timestamp=True)

        # Deletes the nan values
        sato2_clean = sato2[sato2[sato2_track].notna()]
        fio2_clean = fio2[fio2[fio2_track].notna()]
        
        # Creates a new dataframe with timestamp | sato2_value | fio2_value where both values come from the same timestamp
        pre_ri= sato2_clean.merge(fio2_clean, on="Time")

        #Creates the RI dataframe: Timestamp | RI_value
        self.values = {'Timestamp': pre_ri["Time"], 'RI': pre_ri[sato2_track] / pre_ri[fio2_track]} 

#Does the ROX index calculation by dividing oxygen saturation by FiO2.
#Does not require any special handling of missing data as this class is only used when we have the data.
#Does not need to handle multiple possible track names as these are fixed.
