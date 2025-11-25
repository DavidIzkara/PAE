import vitaldb
from vitaldb import VitalFile
import os
import pandas as pd

from check_avalability import check_availability
from shock_index import ShockIndex
from driving_pressure import DrivingPressure
from dynamic_compliance import DynamicCompliance
from rox_index import RoxIndex
from temp_comparison import TempComparison
from cardiac_output import CardiacOutput
from systemic_vascular_resistance import SystemicVascularResistance
from cardiac_power_output import CardiacPowerOutput
from effective_arterial_elastance import EffectiveArterialElastance
from heart_rate_variability import HeartRateVariability

def key_datetime(fname):
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split('_')
    return parts[-2] + parts[-1]

def find_latest_vital(recordings_dir):
    if not os.path.isdir(recordings_dir):
        return None

    # Numeric subfolders
    folders = [d for d in os.listdir(recordings_dir)
               if os.path.isdir(os.path.join(recordings_dir, d)) and d.isdigit()]
    if not folders:
        return None

    latest_folder = sorted(folders)[-1]
    folder_path = os.path.join(recordings_dir, latest_folder)

    # Latest .vital file
    vitals = [f for f in os.listdir(folder_path) if f.endswith('.vital')]
    if not vitals:
        return None

    latest = sorted(vitals, key=key_datetime)[-1]
    return os.path.join(folder_path, latest)


recordings_dir = r"" ##poner el path donde se guarda el archivo
vital_path = find_latest_vital(recordings_dir)
vf = VitalFile(vital_path)

tracks = vf.get_track_names()
#print("Track names: ", tracks)
possible_list = check_availability(tracks)
#print("Los algoritmos disponibles son: ", possible_list)

results = {}
for algorithm in possible_list:
    
    if algorithm == 'Shock Index':
        results['Shock Index'] = ShockIndex(vf).values
        print(type(results['Shock Index']))
    elif algorithm == 'Driving Pressure':
        results['Driving Pressure'] = DrivingPressure(vf).values
    elif algorithm == 'Dynamic Compliance':
        results['Dynamic Compliance'] = DynamicCompliance(vf).values
    elif algorithm == 'ROX Index':
        results['ROX Index'] = RoxIndex(vf).values
    elif algorithm == 'Temp Comparison':
        results['Temp Comparison'] = TempComparison(vf).values
    elif algorithm == 'Cardiac Output':
        results['Cardiac Output'] = CardiacOutput(vf).values
    elif algorithm == 'Systemic Vascular Resistance':
        results['Systemic Vascular Resistance'] = SystemicVascularResistance(vf).values
    elif algorithm == 'Cardiac Power Output':
        results['Cardiac Power Output'] = CardiacPowerOutput(vf).values
    elif algorithm == 'Effective Arterial Elastance':
        results['Effective Arterial Elastance'] = EffectiveArterialElastance(vf).values
    
    '''
    with pd.ExcelWriter("salida.xlsx") as writer:
        for nombre, tabla in results.items():
            df = pd.DataFrame(tabla)
            df.to_excel(writer, sheet_name=nombre[:31], index= False)

    for nombre, tabla in results.items():
        df = pd.DataFrame(tabla)
        df.to_csv(f"{nombre}.csv", index= False)
    '''
