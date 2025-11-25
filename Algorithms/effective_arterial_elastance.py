import vitaldb

class EffectiveArterialElastance:
    def __init__(self, vf: vitaldb.VitalFile):
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

        # Creates a new dataframe with timestamp | hr_value | sys_value where both values come from the same timestamp
        pre_eae= sys_clean.merge(bld_clean, on="Time")

        #Creates the SI dataframe: Timestamp | SI_value
        self.values = {'Timestamp': pre_eae["Time"], 'EAE': (0.9 * pre_eae[sys_track]) / pre_eae[bld_track]} 
