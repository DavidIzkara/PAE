import zarr
from numcodecs import Blosc 
import os
import time
import numpy as np
import pandas as pd

from Zarr.utils_zarr_corrected import STORE_PATH, read_group_zattrs, FRAME_SIGNAL_DEMO, FRAME_SIGNAL, TIMESTAMP_FORMAT, get_track_names_simplified, write_prediction, VISIBLE_ALGORITHMS

DEMO = False

# --- Inicializacion de los algoritmos (los que lo necessitan) ---

#from Algorithms.baroreflex_sensitivity import BaroreflexSensitivity
#BRS = BaroreflexSensitivity()

#from Algorithms.blood_pressure_variability import BloodPressureVariability 
#BPV = BloodPressureVariability()

from Algorithms.heart_rate_variability import HeartRateVariability 
HRV = HeartRateVariability()

from Algorithms.respiratory_sinus_arrhythmia import RespiratorySinusArrhythmia
RSA = RespiratorySinusArrhythmia()

# ----------------------------------------------------------------

last_processed_timestamp = 0.0

def monitorizar_actualizacion_iterativo() -> bool:
    """
    Sondea de forma recursiva una pista y ajusta el intervalo de espera.

    Args:
        track_name: Nombre de la pista a monitorizar.
        diferencia_anterior_segundos: La diferencia de tiempo de la iteración anterior.
        
    Returns:
        True si se detecta una actualización, False si se interrumpe o falla.
    """
    global last_processed_timestamp

    try:
        metadatos = read_group_zattrs(STORE_PATH, "")
        last_updated_str = metadatos.get('last_updated')

        if not last_updated_str:
            return False

        archivo_ts = time.mktime(time.strptime(last_updated_str, TIMESTAMP_FORMAT))

        if archivo_ts > last_processed_timestamp:
            print(f"\n⚡ Cambio detectado: Archivo({last_updated_str}) > Procesado anteriormente")
            last_processed_timestamp = archivo_ts
            return True

        return False
    
    except Exception as e:
        return False
    """
    diferencia_anterior_segundos = 10000 # Inicialmente un valor alto, para poder detectar el canvio
    
    while True:
        try:
            # Obtener y convertir el timestamp
            metadatos = read_group_zattrs(STORE_PATH, "")
            last_updated_str = metadatos.get('last_updated')
            

            if not last_updated_str:
                print(f"--- [ERROR] Clave 'last_updated' no encontrada. Esperando 5s y reintentando...")
                time.sleep(5.0)

            # Convertir la cadena a objeto datetime y calcular la diferencia
            diferencia_tiempo = time.mktime(time.strptime(time.strftime(TIMESTAMP_FORMAT), TIMESTAMP_FORMAT)) - time.mktime(time.strptime(last_updated_str, TIMESTAMP_FORMAT))
            print(f"--- Tiempo sin actualizar: {diferencia_tiempo:.2f} s. ", end="")


            # Comprobar si hay una actualización
            # Si la diferencia actual es menor que la anterior, significa que 'last_updated' se actualizó.
            if diferencia_tiempo < diferencia_anterior_segundos:
                if diferencia_anterior_segundos != 10000: # Ignorar la primera llamada
                    # ¡ACTUALIZACIÓN DETECTADA!
                    print(f"\n--- ¡ACTUALIZACIÓN DETECTADA! ({diferencia_anterior_segundos:.2f}s -> {diferencia_tiempo:.2f}s)")
                    return True # Señal de éxito: Propaga True hacia arriba.

            # Determinar el intervalo de espera (tiempo de 'sleep')
            sleep_time = 0.5 # Al final lo dejamos en 0.5 pa que no se quede pillado aqui
                
            # Actualizar el valor anterior y pausar
            print(f"--- Próxima verificación en {sleep_time} s...")
            diferencia_anterior_segundos = diferencia_tiempo
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n--- Monitoreo detenido por el usuario. ---")
            return False
        except Exception as e:
            print(f"\n--- [ERROR CRÍTICO] Ocurrió un error: {e}. Reintentando en 5s.")
            time.sleep(5.0)
            # En un error crítico, reiniciamos la "diferencia anterior"
            diferencia_anterior_segundos = 10000 
    """
    
def leer_ultimas_muestras_zarr(zarr_path: str, sample: str, last_samples: int) -> pd.DataFrame:
    """
    Recupera las últimas 'last_samples' muestras (time_ms y value) de una pista
    específica dentro del Zarr y las devuelve como un DataFrame.

    Args:
        zarr_path: La ruta completa al archivo .zarr (el path del root).
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
            print(f"--- [ERROR] La pista '{sample}' no se encuentra en Zarr en la ruta '{sample}'.")
            return empty_df
            
        grp = root[sample]
        
        # Verificar que los datasets 'time_ms' y 'value' existan
        if "time_ms" not in grp or "value" not in grp:
            print(f"--- [ERROR] Datasets 'time_ms' o 'value' faltan en el grupo '{sample}'.")
            return empty_df
            
        ds_time = grp["time_ms"]
        ds_val = grp["value"]
        
        # Determinar la longitud total de la pista
        total_samples = ds_time.shape[0]
        
        if total_samples == 0:
            return empty_df
            
        # Calcular el índice de inicio para el slicing
        start_index = max(0, total_samples - last_samples)
        
        # Aplicar el slicing para leer las N últimas muestras
        time_slice = ds_time[start_index:]
        value_slice = ds_val[start_index:]
        
        # Crear el DataFrame
        df = pd.DataFrame({
            'time_ms': time_slice,
            'value': value_slice
        })
        
        return df

    except FileNotFoundError:
        print(f"--- [ERROR] El archivo Zarr no se encontró en la ruta: {zarr_path}")
        return empty_df
    except Exception as e:
        print(f"--- [ERROR CRÍTICO] Ocurrió un error al leer Zarr: {e}")
        return empty_df


def main_to_loop(algoritmes_escollits):
    try:
        #print(f"--- Iniciando Monitoreo Recursivo para los Tracks ---")

        if monitorizar_actualizacion_iterativo(): # Esto es un await de toda la vida
            print("\n--- ¡ACTUALIZACIÓN DETECTADA !")
            
            metadatos_general = read_group_zattrs(STORE_PATH, "") # Leer el .zattrs del root
            tracks = get_track_names_simplified(STORE_PATH) # Obtener nombre de variables del Zarr (sean actualizados o no, aun no podemos distingirlos)

            tracks_updated = []
            dataframes = {}
        
            if DEMO:
                Frame = FRAME_SIGNAL_DEMO
            else:
                Frame = FRAME_SIGNAL
            
            for track in tracks:
                metadatos_track = read_group_zattrs(STORE_PATH, Frame + track) # Leer el .zattrs de la varible concreta
                if metadatos_track is not None:
                    if metadatos_track.get("last_updated") == metadatos_general.get("last_updated"): # Comparar la ultima actualizacion, para saber si se ha actualizado la variable o no 
                        print(f"--- La track '{track}' fue actualizada ({metadatos_track.get('last_update_secs')} muestras).")
                        #print(f"        - Ultima actualización contiene datos de {metadatos_track.get('last_update_secs')} segundos que se representan en {metadatos_track.get('last_update_samples')} samples")
                        sample = Frame + track # Juntar el path completo del track dentro del zarr
                        df_track = leer_ultimas_muestras_zarr(STORE_PATH, sample, metadatos_track.get('last_update_samples')) # Leer las ultimas muestras del zarr
                        dataframes[sample.removeprefix("signals/")] = df_track # Guardar el dataframe en un diccionario (Lista de dataframes), quitando el prefijo "signals/"
                        #print(f"        - Se guarda el dataframe en la lista.")
                        tracks_updated.append(track) # Guardar el nombre de la variable actualizada
                    else:
                        print(f"--- La track '{track}' NO fue actualizada.")
                    
            print(f"--- Todos los dataframes actualizados") 
            print(f"--- Lista de variables recogidas: {tracks_updated}")
            # list_available = check_availability('Intellivue/' + tracks_updated) # Comprobar que algoritmos se pueden calcular con las variables actualizadas
            print(f"--- Algoritmos que se puedan calcular: {algoritmes_escollits}")
            
            results = {}

            for algoritme in algoritmes_escollits: # Por cada algoritmo disponible, importarlo i calcularlo, menys els que tenen buffers
                match algoritme:
                    #case 'BRS':
                        #results['BRS'] = BRS.compute(dataframes)
                    case 'Blood Pressure Variability':
                        from Algorithms.blood_pressure_variability import BloodPressureVariability      # Comentar esto y descomentar las lineas del BPV del inicio del doc
                        results['Blood Pressure Variability'] = BloodPressureVariability(dataframes).values # Comentar esto y descomentar siguiente linea
                        #results['Blood Pressure Variability'] = BPV.compute(dataframes)
                    case 'Cardiac Output':
                        from Algorithms.cardiac_output import CardiacOutput
                        results['Cardiac Output'] = CardiacOutput(dataframes).values
                    case 'Cardiac Power Output':
                        from Algorithms.cardiac_power_output import CardiacPowerOutput
                        results['Cardiac Power Output'] = CardiacPowerOutput(dataframes).values
                    case 'Driving Pressure':
                        from Algorithms.driving_pressure import DrivingPressure
                        results['Driving Pressure'] = DrivingPressure(dataframes).values
                    case 'Dynamic Compliance':
                        from Algorithms.dynamic_compliance import DynamicCompliance
                        results['Dynamic Compliance'] = DynamicCompliance(dataframes).values
                    case 'Effective Arterial Elastance':   
                        from Algorithms.effective_arterial_elastance import EffectiveArterialElastance
                        results['Effective Arterial Elastance'] = EffectiveArterialElastance(dataframes).values
                    case 'Heart Rate Variability':
                        results['Heart Rate Variability'] = HRV.compute(dataframes)
                    case 'RSA':
                        results['RSA'] = RSA.compute(dataframes)
                    case 'ROX Index':
                        from Algorithms.rox_index import RoxIndex
                        results['ROX Index'] = RoxIndex(dataframes).values
                    case 'Shock Index':
                        from Algorithms.shock_index import ShockIndex
                        results['Shock Index'] = ShockIndex(dataframes).values
                    case 'Systemic Vascular Resistance':
                        from Algorithms.systemic_vascular_resistance import SystemicVascularResistance
                        results['Systemic Vascular Resistance'] = SystemicVascularResistance(dataframes).values
                    case 'Temp Comparison':
                        from Algorithms.temp_comparison import TempComparison
                        results['Temp Comparison'] = TempComparison(dataframes).values
                    case _:
                        print(f"--- Advertencia: Algoritmo '{algoritme}' no encontrado")
                        pass
            print(f"--- Resultados de los algoritmos: {results}")
            return results
        else:
            return None # En caso que falle el monitoreo o se interrumpa

    except KeyboardInterrupt:
        print("\n--- Monitoreo detenido por el usuario. ---")
        return None

if __name__ == "__main__":
    try:
        while True: # Esto se deberia hacer en el main, para poder devolver los resultados al front
            results = main_to_loop() # Esperar a que se detecte una actualización y obtener los resultados de los algoritmos.

            for Nombre_algoritmo, df_result in results.items():
                value_columns = [col for col in df_result.columns if col != 'Timestamp' and col != 'Time_ini_ms' and col != 'Time_fin_ms']
                time_ms_array = df_result['Timestamp'].values
                
                visible = False # Bool que dicta si se puede enseñar la función
                if Nombre_algoritmo in VISIBLE_ALGORITHMS: # Depende de si aparece en la lista de algoritmos visibles (Zarr/utils_zarr_corrected.py)
                    visible = True
                
                for track_name in value_columns:
                    value_array = df_result[track_name].values
                    write_prediction(STORE_PATH, track_name, time_ms_array, value_array, modelo_info={"model": Nombre_algoritmo, "visibilidad": visible})

    except KeyboardInterrupt:
        print("\n--- Monitoreo detenido por el usuario. ---")
