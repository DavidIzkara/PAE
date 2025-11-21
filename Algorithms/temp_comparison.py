import vitaldb

class TempComparison:
    def __init__(self, vf: vitaldb.VitalFile):

        available_tracks = vf.get_track_names()

        # Try to find core and skin temperature tracks
        core_track = next(
            (t for t in available_tracks if 'Intellivue/BT_CORE' in t), # First try for BT_CORE
            next((t for t in available_tracks if 'Intellivue/BT_BLD' in t), None)) # Then try for generic BT_BLD track
        skin_track = next(
            (t for t in available_tracks if 'Intellivue/BT_SKIN' in t), # First try for BT_SKIN
            next((t for t in available_tracks if 'Intellivue/TEMP' in t), None)) # Then try for generic TEMP track
        

        # Converts the signals to pandas dataframes
        core = vf.to_pandas(track_names=core_track, interval=0, return_timestamp=True)
        skin = vf.to_pandas(track_names=skin_track, interval=0, return_timestamp=True)

        
        # Deletes the nan values
        core_clean = core[core[core_track].notna()]
        skin_clean = skin[skin[skin_track].notna()]
        
        # Creates a new dataframe with timestamp | core_value | skin_value where both values come from the same timestamp
        pre_temp= core_clean.merge(skin_clean, on="Time")

        #Creates the SI dataframe: Timestamp | core_value | skin_value
        self.values = {'Timestamp': pre_temp["Time"], 'CORE': pre_temp[core_track] , 'SKIN': pre_temp[skin_track]} 
