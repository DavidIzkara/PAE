import vitaldb
import pandas as pd

class ShockIndex: 

    def __init__(self, vf: vitaldb.VitalFile):
        # Get all available track names in the VitalFile
        available_tracks = vf.get_track_names()

        # Try to find heart rate and systolic pressure tracks
        hr_track = next(
            (t for t in available_tracks if 'Intellivue/ECG_HR' in t),          # First try for ECG_HR
            next((t for t in available_tracks if 'Intellivue/ABP_HR' in t),     # Then try for ABP_HR
                 next((t for t in available_tracks if 'Intellivue/HR' in t), None))) # Finally try for generic HR track
        sys_track = next(
            (t for t in available_tracks if 'Intellivue/ABP_SYS' in t), # First try for invasive systolic BP
            next((t for t in available_tracks if 'Intellivue/BP_SYS' in t), # Then try for another possible invasive systolic BP
                next((t for t in available_tracks if 'Intellivue/NIBP_SYS' in t), None))) # Finally try for non-invasive systolic BP
        
        # Converts the signals to pandas dataframes
        hr = vf.to_pandas(track_names=hr_track, interval=0, return_timestamp=True)
        sys = vf.to_pandas(track_names=sys_track, interval=0, return_timestamp=True)

        
        # Deletes the nan values
        hr_clean = hr[hr[hr_track].notna()]
        sys_clean = sys[sys[sys_track].notna()]
        
        # Creates a new dataframe with timestamp | hr_value | sys_value where both values come from the same timestamp
        pre_si= hr_clean.merge(sys_clean, on="Time")

        pd.set_option('display.float_format', '{:.0f}'.format)

        print("Timestamp", pre_si.head(n=5))
        #Creates the SI dataframe: Timestamp | SI_value
        self.values = {'Timestamp': pre_si["Time"], 'SI': pre_si[hr_track] / pre_si[sys_track]} 

#Handles both invasive and non-invasive blood pressure by checking available tracks.
#Requires heart rate track, tries multiple possible names for robustness.
#Does not require any special handling of missing data as this class is only used when we have the data.
