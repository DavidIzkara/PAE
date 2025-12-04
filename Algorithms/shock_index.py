import vitaldb
import pandas as pd

class ShockIndex: 

    def __init__(self, data):

        if isinstance(data, vitaldb.VitalFile):
            self._from_vf(data)
        else:
            self._from_df(data)

    def _from_vf(self, vf):
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

        print("Timestamp", pre_si.head(n=5))
        #Creates the SI dataframe: Timestamp | SI_value
        self.values = pd.DataFrame({'Timestamp': pre_si["Time"], 'SI': pre_si[hr_track] / pre_si[sys_track]})


    def _from_df(self, list_dataframe: list[pd.DataFrame]):
        #Se recibe una lista de dataframes
        #Se sacan los indices, que son los nombres de las variables
        available_tracks = list_dataframe.keys()

        # Try to find heart rate tracks
        hr_track = next(
            (t for t in available_tracks if 'Intellivue/ECG_HR' in t),          # First try for ECG_HR
            next((t for t in available_tracks if 'Intellivue/ABP_HR' in t),     # Then try for ABP_HR
                    next((t for t in available_tracks if 'Intellivue/HR' in t), None))) # Finally try for generic HR track
        sys_track = next(
            (t for t in available_tracks if 'Intellivue/ABP_SYS' in t), # First try for invasive systolic BP
            next((t for t in available_tracks if 'Intellivue/BP_SYS' in t), # Then try for another possible invasive systolic BP
                next((t for t in available_tracks if 'Intellivue/NIBP_SYS' in t), None))) # Finally try for non-invasive systolic BP

        hr = list_dataframe[hr_track] 
        sys = list_dataframe[sys_track]
        
        # Deletes the nan values
        hr_clean = hr[hr["value"].notna()]
        sys_clean = sys[sys["value"].notna()]

        # Creates a new dataframe with timestamp | hr_value | sys_value where both values come from the same timestamp
        pre_si= hr_clean.merge(sys_clean, on="time_ms")
        #print(pre_si)

        #Creates the SI dataframe: Timestamp | SI_value
        self.values = pd.DataFrame({'Timestamp': pre_si["time_ms"], 'SI': pre_si["value_x"] / pre_si["value_y"]})
       

#Handles both invasive and non-invasive blood pressure by checking available tracks.
#Requires heart rate track, tries multiple possible names for robustness.
#Does not require any special handling of missing data as this class is only used when we have the data.
