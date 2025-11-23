import vitaldb
from vitaldb import VitalFile
import os
import zarr
from utils_zarr import leer_multiples_senyales, leer_senyal, vital_to_zarr

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


#####           PRUEBAS VF                      ########

#recordings_dir = r"C:\Users\doi99\Desktop\PAE\records"
#vital_path = find_latest_vital(recordings_dir)
#vf = VitalFile(vital_path)

#tracks = vf.get_track_names()
#print("Track names: ", tracks)
#possible_list = check_availability(tracks)
#print("Los algoritmos disponibles son: ", possible_list )

#results = {}
#for algorithm in possible_list:
    '''
    if algorithm == 'Shock Index':
        results['Shock Index'] = ShockIndex(vf).values
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
   # if algorithm == 'Heart Rate Variability':
   #     results['Heart Rate Variability'] = HeartRateVariability(vf)  #No tiene values definido aún
    
    #Pendiente añadir Variables autonomicas
    #elif algorithm == 'ICP Model':
    #    results['ICP Model'] = icp_model() #Pendiente ver como añadir el modelo de ICP
    #elif algorithm == 'ABP Model':
    #    results['ABP Model'] = abp_model() #Pendiente ver como añadir el modelo de ABP

    #Pendiente añadir otros algoritmos.

#Print results to verify
#print("DC: ", results['Dynamic Compliance'])

#Pendientes de verificar con otro .vital
#print("RI: ", results['ROX Index'])
#print("TEMP: ", results['Temp Comparison'])
#print("CO: ", results['Cardiac Output'])
#print("SRV: ", results['Systemic Vascular Resistance'])    PDT REVISAR que funcione el merge.
#print("CPO: ", results['Cardiac Power Output'])            PDT REVISAR que funcione el merge.
#print("EAE: ", results['Effective Arterial Elastance'])

#####           PRUEBAS ZARR                      ########

#zarr_Path = r"C:\Users\doi99\Desktop\PAE\records\test_alg.zarr"
df = leer_senyal(
    zarr_path=r"C:\Users\doi99\Desktop\PAE\records",
    track="signals/Intellivue/ECG_HR"
)
print("porfavor:", df)


#Función para crear un fucking zarr

#recordings_dir = r"C:\Users\doi99\Desktop\PAE\records"
#vital_path = find_latest_vital(recordings_dir)
#vf = VitalFile(vital_path)
#tracks = vf.get_track_names()

#zarr_path =r"C:\Users\doi99\Desktop\PAE\records"
#vital_to_zarr( vital_path, zarr_path, tracks)


zarr_path=r"C:\Users\doi99\Desktop\PAE\records"
tracks = ['Intellivue/ECG_HR', 'Intellivue/ABP_HR', 'Intellivue/HR', 'Intellivue/ABP_SYS', 'Intellivue/BP_SYS', 'Intellivue/NIBP_SYS']
dataframe = leer_multiples_senyales(zarr_path,tracks)
print("DF: ", dataframe)
si = ShockIndex(dataframe).values
print("DATAFRAME de SI:", si)


#recordings_dir = r"C:\Users\doi99\Desktop\PAE\records"
#vital_path = find_latest_vital(recordings_dir)
#vf = VitalFile(vital_path)
