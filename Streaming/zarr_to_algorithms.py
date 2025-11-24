import zarr
from numcodecs import Blosc 
import os
import time
import numpy as np
import pandas as pd

from utils_Streaming import OUTPUT_DIR
from utils_zarr import leer_zattrs_de_grupo, FRAME_SIGNAL_DEMO, FRAME_SIGNAL, FORMATO_TIMESTAMP, get_track_names_simplified, escribir_prediccion, ALGORITMOS_VISIBLES
from check_availability import check_availability

#from Algorithms.blood_pressure_variability import BloodPressureVariability 
from Algorithms.cardiac_output import CardiacOutput
from Algorithms.cardiac_power_output import CardiacPowerOutput
from Algorithms.driving_pressure import DrivingPressure
from Algorithms.dynamic_compliance import DynamicCompliance
from Algorithms.effective_arterial_elastance import EffectiveArterialElastance
#from Algorithms.heart_rate_variability import HeartRateVariability 
from Algorithms.rox_index import RoxIndex
from Algorithms.shock_index import ShockIndex
from Algorithms.systemic_vascular_resistance import SystemicVascularResistance
from Algorithms.temp_comparison import TempComparison

DEMO = False

ZARR_PATH = os.path.join(OUTPUT_DIR, "session_data.zarr") # Ruta al archivo Zarr de ejemplo

def monitorizar_actualizacion_recursivo(
    diferencia_anterior_segundos: float = 10000
) -> bool:
    """
    Sondea de forma recursiva una pista y ajusta el intervalo de espera.

    Args:
        track_name: Nombre de la pista a monitorizar.
        diferencia_anterior_segundos: La diferencia de tiempo de la iteración anterior.
        
    Returns:
        True si se detecta una actualización, False si se interrumpe o falla.
    """
    try:
        # Obtener y convertir el timestamp
        metadatos = leer_zattrs_de_grupo(ZARR_PATH, "")
        last_updated_str = metadatos.get('last_updated')

        if not last_updated_str:
            print(f"[ERROR] Clave 'last_updated' no encontrada. Esperando 5s y reintentando...")
            time.sleep(5.0)
            return monitorizar_actualizacion_recursivo(diferencia_anterior_segundos) # Volver a llamar

        # Convertir la cadena a objeto datetime y calcular la diferencia
        diferencia_tiempo = time.mktime(time.strptime(time.strftime(FORMATO_TIMESTAMP), FORMATO_TIMESTAMP)) - time.mktime(time.strptime(last_updated_str, FORMATO_TIMESTAMP))
        print(f"Tiempo sin actualizar: {diferencia_tiempo:.2f} s. ", end="")


        # Comprobar si hay una actualización
        # Si la diferencia actual es menor que la anterior, significa que 'last_updated' se actualizó.
        if diferencia_tiempo < diferencia_anterior_segundos:
            if diferencia_anterior_segundos != 10000: # Ignorar la primera llamada
                 # ¡ACTUALIZACIÓN DETECTADA!
                 print(f"\n   ¡ACTUALIZACIÓN DETECTADA! ({diferencia_anterior_segundos:.2f}s -> {diferencia_tiempo:.2f}s)")
                 return True # Señal de éxito: Propaga True hacia arriba.

        # Determinar el intervalo de espera (tiempo de 'sleep')
        if 0 <= diferencia_tiempo <= 10:
            sleep_time = 5.0
        elif 10 < diferencia_tiempo <= 20:
            sleep_time = 2.0
        elif 20 < diferencia_tiempo <= 25:
            sleep_time = 1.0
        else: # diferencia_tiempo > 25
            sleep_time = 0.5
            
        # Actualizar el valor anterior y pausar
        print(f"Próxima verificación en {sleep_time} s...")
        time.sleep(sleep_time)

        # Llamada Recursiva
        # Si la llamada recursiva retorna True, nosotros también retornamos True.
        if monitorizar_actualizacion_recursivo(diferencia_tiempo):
            return True
        
        return False # Esto se alcanzaría si una rama de la recursión se agota o se detiene. (La recursividad devuelve False, nosotros tambien tenemos que hacerlo)

    except RecursionError:
        print("\n[PELIGRO] Se alcanzó el límite de profundidad de recursión de Python. Deteniendo el monitoreo.")
        return False
    except KeyboardInterrupt:
        print("\n--- Monitoreo detenido por el usuario. ---")
        return False
    except Exception as e:
        print(f"\n[ERROR CRÍTICO] Ocurrió un error: {e}. Deteniendo el monitoreo.")
        return False
    


def leer_ultimas_muestras_zarr(
    zarr_path: str, 
    sample: str, 
    last_samples: int
) -> pd.DataFrame:
    """
    Recupera las últimas 'last_samples' muestras (time_ms y value) de una pista
    específica dentro del contenedor Zarr y las devuelve como un DataFrame.

    Args:
        zarr_path: La ruta completa al archivo .zarr (el contenedor raíz).
        track: El nombre de la pista (ej. "signals/Intelliuve/ECG_HR").
        last_samples: El número N de muestras a recuperar desde el final del array.

    Returns:
        Un pandas.DataFrame con las columnas ['time_ms', 'value'].
        Si el track no existe o está vacío, devuelve un DataFrame vacío.
    """
    
    # Definir la estructura del DataFrame vacío para manejar errores o datos no encontrados
    empty_df = pd.DataFrame(columns=['time_ms', 'value'])
    empty_df['time_ms'] = empty_df['time_ms'].astype(np.int64)
    empty_df['value'] = empty_df['value'].astype(np.float32)

    try:
        # Abrir el contenedor Zarr en modo sólo lectura
        root = zarr.open(zarr_path, mode='r')
        
        # Acceder al grupo de la pista
        if sample not in root:
            print(f"[ERROR] La pista '{sample}' no se encuentra en Zarr en la ruta '{sample}'.")
            return empty_df
            
        grp = root[sample]
        
        # Verificar que los datasets 'time_ms' y 'value' existan
        if "time_ms" not in grp or "value" not in grp:
            print(f"[ERROR] Datasets 'time_ms' o 'value' faltan en el grupo '{sample}'.")
            return empty_df
            
        ds_time = grp["time_ms"]
        ds_val = grp["value"]
        
        # 1. Determinar la longitud total de la pista
        total_samples = ds_time.shape[0]
        
        if total_samples == 0:
            return empty_df
            
        # 2. Calcular el índice de inicio para el slicing
        start_index = max(0, total_samples - last_samples)
        
        # 3. Aplicar el slicing para leer las N últimas muestras
        time_slice = ds_time[start_index:]
        value_slice = ds_val[start_index:]
        
        # 4. Crear el DataFrame
        df = pd.DataFrame({
            'time_ms': time_slice,
            'value': value_slice
        })
        
        return df

    except FileNotFoundError:
        print(f"[ERROR] El archivo Zarr no se encontró en la ruta: {zarr_path}")
        return empty_df
    except Exception as e:
        print(f"[ERROR CRÍTICO] Ocurrió un error al leer Zarr: {e}")
        return empty_df


# Ejemplo de uso de la función recursiva
def main_to_loop():
    try:
        # traks_a_monitorizar = ["ECG", "NIBP_DBP", "SPO2_SAT"] # Ejemplo de tracks 

        print(f"--- Iniciando Monitoreo Recursivo para los Tracks ---")

        if monitorizar_actualizacion_recursivo():
            print("\n¡ACTUALIZACIÓN DETECTADA !")
            
            metadatos_general = leer_zattrs_de_grupo(ZARR_PATH, "")
            tracks = get_track_names_simplified(ZARR_PATH)

            tracks_updated = []
            dataframes = {}
        
            if DEMO:
                Frame = FRAME_SIGNAL_DEMO
            else:
                Frame = FRAME_SIGNAL
            
            for track in tracks:
                metadatos_track = leer_zattrs_de_grupo(ZARR_PATH, Frame + track)
                if metadatos_track is not None:
                    if metadatos_track.get("last_updated") == metadatos_general.get("last_updated"):
                        print(f"   - La track '{track}' fue actualizada.")
                        print(f"        - Ultima actualización contiene datos de {metadatos_track.get('last_update_secs')} segundos que se representan en {metadatos_track.get('last_update_samples')} samples")
                        sample = Frame + track
                        df_track = leer_ultimas_muestras_zarr(ZARR_PATH, sample, metadatos_track.get('last_update_samples'))
                        dataframes[sample.removeprefix("signals/")] = df_track
                        print(f"        - Se guarda el dataframe en la lista.")
                        tracks_updated.append(track)
                    else:
                        print(f"   - La track '{track}' NO fue actualizada.")
                    
            print(f" - Todos los dataframes actualizados") 
            print(f" Lista de variables recogidas: {tracks_updated}")
            list_available = check_availability(tracks_updated)
            print(f" Algoritmos que se puedan calcular: {list_available}")
            
            results = {}

# Falten per importar: La llibreria dels BPV i HRV (falta la llibreria aquella, ecgdetectors)

            for algoritme in list_available:
                match algoritme:
                    case 'Shock Index':
                        results['Shock Index'] = ShockIndex(dataframes).values
                    case 'Driving Pressure':
                        results['Driving Pressure'] = DrivingPressure(dataframes).values
                    case 'Dynamic Compliance':
                        results['Dynamic Compliance'] = DynamicCompliance(dataframes).values
                    case 'ROX Index':
                        results['ROX Index'] = RoxIndex(dataframes).values
                    case 'Temp Comparison':
                        results['Temp Comparison'] = TempComparison(dataframes).values
                    case 'Cardiac Output':
                        results['Cardiac Output'] = CardiacOutput(dataframes).values
                    case 'Systemic Vascular Resistance':
                        results['Systemic Vascular Resistance'] = SystemicVascularResistance(dataframes).values
                    case 'Cardiac Power Output':
                        results['Cardiac Power Output'] = CardiacPowerOutput(dataframes).values
                    case 'Effective Arterial Elastance':
                        results['Effective Arterial Elastance'] = EffectiveArterialElastance(dataframes).values
                    # case 'Blood Pressure Variability':
                    #     results['Blood Pressure Variability'] = BloodPressureVariability(dataframes).values
                    # case 'Heart Rate Variability':
                    #     results['Heart Rate Variability'] = HeartRateVariability(dataframes).values
                    case _:
                        print(f"Advertencia: Algoritmo '{algoritme}' no encontrado")
                        pass
            print(f"Resultados de los algoritmos: {results}")
            return results
                    
    except KeyboardInterrupt:
        print("\n--- Monitoreo detenido por el usuario. ---")

if __name__ == "__main__":
    try:
        while True:
            results = main_to_loop()

            for Nombre_algoritmo, df_result in results.items():
                value_columns = [col for col in df_result.columns if col != 'Timestamp']
                time_ms_array = df_result['Timestamp'].values
                
                visible = False # Bool que dicta si se puede enseñar la función
                if Nombre_algoritmo in ALGORITMOS_VISIBLES:
                    visible = True
                
                for track_name in value_columns:
                    value_array = df_result[track_name].values
                    escribir_prediccion(ZARR_PATH, track_name, time_ms_array, value_array, modelo_info={"model": Nombre_algoritmo, "visibilidad": visible})

    except KeyboardInterrupt:
        print("\n--- Monitoreo detenido por el usuario. ---")
