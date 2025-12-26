import os
import time
import numpy as np
import threading
import random 
import pandas as pd
import vitaldb
from Algorithms.check_availability import check_availability
from Zarr.utils_zarr_corrected import VISIBLE_ALGORITHMS, _DEFAULT_COMPRESSOR, STORE_PATH, safe_group, get_group_if_exists, append_1d, open_root
from Streaming.utils_Streaming import WAVE_TRACKS_FREQUENCIES, WAVE_STANDARD_RATE, obtener_vital_timestamp, obtener_directorio_del_dia, obtener_vital_mas_reciente


BASE_DIR = r"C:\Users\UX636EU\OneDrive - EY\Desktop\recordings" 
POLLING_INTERVAL = 1 

# -------------------------------------------------------------------------------------------
PRUEVAS = True

DIRECTORIO_PRUEVA = r"C:\Users\UX636EU\Downloads" 
ARCHIVO_VITAL = r"xxycag2xd_250512_074519" 

SIM_MIN_SECS = 20
SIM_MAX_SECS = 30

# -------------------------------------------------------------------------------------------

def vital_to_zarr_streaming(
    vital_path: str,
    available_tracks: list,
    zarr_path: str,
    last_read_counts: dict, 
    simulated_growth_seconds: float | None = None,
    chunk_len: int = 30000,
) -> dict: 
    """
    Processa l'arxiu vital utilitzant la lògica de freqüències i el slicing per 
    conteig de mostres (last_read_counts) en mode simulació.
    """

    vf = vitaldb.VitalFile(vital_path)
    #available_tracks = vf.get_track_names() # Recoje las cabezeras del vitalfile (nombre de las variables)

    #os.makedirs(os.path.dirname(zarr_path) or ".", exist_ok=True)
    root = open_root(zarr_path)

    signals_root = safe_group(root, "signals") # Abre o crea el directorio de signals dentro del root

    written_any = False     # Creacion de variables que vamos a utilizar
    total_added_samples = 0
    written_tracks = 0
    skipped_no_new = 0
    skipped_empty = 0
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Copiamos el last_read_count (sera una mikyherramienta que usaremos mas tarde)
    new_last_read_counts = last_read_counts.copy() 
    tracks_updated = []

    for track in available_tracks:  # Para cada variable hacer todo lo siguiente
        rate = 1.0 
        interval = 1.0  # Establecemos los valores por defecto de los Nominales
        track_type = "NUM"
        
        # En caso de que las variables existan dentro de la WAVE_TRACKS_FREQUENCIES (importado de utils_Streaming.py)
        if track in WAVE_TRACKS_FREQUENCIES:
            track_type = "WAVE"
            rate_configurada = WAVE_TRACKS_FREQUENCIES.get(track, 0)    # Reconfiguramos los parametros para que sean los correctos para los WAVE
            if rate_configurada > 0:
                rate = rate_configurada 
            else:
                rate = WAVE_STANDARD_RATE # Si la rate no esta configurada en el import, usar la STANDARD (100 Hz)
            interval = 1.0 / rate 

        # Leer la variable a la frequencia determinada (para evitar valores vacios o NaN)
        try:
            data = vf.to_numpy([track], interval=interval, return_timestamp=True)
        except Exception as e:
            print(f"-- [WARN] No s'ha pogut llegir el track '{track}' amb interval={interval}: {e}")
            skipped_empty += 1
            continue

        if data is None or data.size == 0:
            skipped_empty += 1
            continue

        ts = data[:, 0].astype(np.float64, copy=False) # Guardar y transformar el timestamp en float64

        raw_vals = data[:, 1]
        vals = np.full(raw_vals.size, np.nan, dtype=np.float64) # Guardar y transoformar los datos en float64

        for i, val in enumerate(raw_vals):
            try:
                vals[i] = float(val)
            except (TypeError, ValueError):
                vals[i] = np.nan

        current_total_count = len(vals) # Guardamos el total de muestras
        last_count = last_read_counts.get(track, 0)
        end_index = current_total_count # Aqui tambien, pero esta variable se usa para el PRUEVAS
        
        # Logica para el Slicing (Formato PRUEVAS)
        if simulated_growth_seconds is not None and simulated_growth_seconds > 0: 
            
            simulated_added_samples = int(simulated_growth_seconds * rate) 
            target_count = min(last_count + simulated_added_samples, current_total_count) 
            
            if target_count <= last_count: 
                skipped_no_new += 1
                new_last_read_counts[track] = last_count # Mantiene el contador antiguo en caso de no tener nuevos datos
                continue
                
            end_index = target_count # Nuevo inice para el slicing

        # Aplica el slicing a los datos (des de last_count hasta end_index)
        ts_slice = ts[last_count:end_index] 
        vals_slice = vals[last_count:end_index]
        
        if ts_slice.size == 0:
            skipped_no_new += 1
            new_last_read_counts[track] = last_count # Mantiene el contador antiguo en caso de no tener nuevos datos
            continue

        # Actualitzar el conteo de mostres para el proximo loop
        new_last_read_counts[track] = end_index 
        
        # Convertir a ms y float32
        ts_ms = np.rint(ts_slice * 1000.0).astype(np.int64)
        vals_f32 = vals_slice.astype(np.float32)

        
        # Checkea que no haya datos en el mismo timestamp, para que no haya duplicacion de este (que los timestamps sean unicos)
        track_group_path = f"signals/{track}"
        existing_grp = get_group_if_exists(root, track_group_path)
        
        if existing_grp is not None and "time_ms" in existing_grp and existing_grp["time_ms"].shape[0] > 0:
            try:
                last_ts_zarr = int(existing_grp["time_ms"][-1])
                mask_new_zarr = ts_ms > last_ts_zarr
                ts_ms = ts_ms[mask_new_zarr]
                vals_f32 = vals_f32[mask_new_zarr]
                
                if ts_ms.size == 0:
                    skipped_no_new += 1
                    continue
                    
            except Exception as e:
                print(f"-- [WARN] No s'ha pogut llegir l'últim time_ms de '{track}' per a deduplicació: {e}")
                pass
                
        # Crea el directorio que contendra los datos de la variable
        grp = safe_group(signals_root, track) 

        try:
            if "time_ms" in grp and "value" in grp:
                ds_time = grp["time_ms"] # Si existe el dataset simplemente guarda el grupo
                ds_val = grp["value"]
            else: # Si no existe el dataset, crealo
                ds_time = grp.require_dataset("time_ms", shape=(0,), maxshape=(None,), chunks=(chunk_len,), dtype="int64", compressor=_DEFAULT_COMPRESSOR)
                ds_val = grp.require_dataset("value", shape=(0,), maxshape=(None,), chunks=(chunk_len,), dtype="float32", compressor=_DEFAULT_COMPRESSOR)
        except Exception as e:
            raise Exception(f"Falla la creació o el resize de l'array Zarr per a '{track}': {type(e).__name__}: {e}")

        append_1d(ds_time, ts_ms) # Sube los datos al dataset (en formato append, para no crearlo de 0)
        append_1d(ds_val, vals_f32)

        grp.attrs["track"] = track
        grp.attrs["sampling_rate_hz"] = rate    # Variables del .zattrs de la Variable
        grp.attrs["track_type"] = track_type 
        grp.attrs["last_updated"] = timestamp
        grp.attrs["total_samples"] = ds_time.shape[0]
        grp.attrs["duration_seconds"] = ds_time.shape[0] / rate if rate > 0 else 0.0
        grp.attrs["last_update_samples"] = ts_ms.size
        
        #print(f"[{track}: {track_type} {rate:.1f} Hz] +{ts_ms.size} mostres (total={ds_time.shape[0]})")
        total_added_samples += int(ts_ms.size)
        tracks_updated.append(track)
        written_tracks += 1
        written_any = True

    print(f"-- Todas las variables actualizadas: {tracks_updated}")
    root.attrs.setdefault("schema", "v1")
    root.attrs.setdefault("created_by", "Streaming")    # Variables del .zattrs del .zarr
    root.attrs.setdefault("time_origin", "epoch1700_ms")
    root.attrs["last_updated"] = timestamp

    if written_any:
        print(f"\n-- Escrita/actualitzada la col·lecció a: {zarr_path}")
        print(f"-- Tracks actualitzades: {written_tracks}, mostres afegides: {total_added_samples}")
    else:
        print(f"\n-- No s'ha escrit cap mostra nova.")
        
    return new_last_read_counts # Devuelve los conteos nuevos


def verificar_y_procesar(vital_path, available_list, last_size, last_read_counts, simulated_growth_seconds):
    """
    Comprova la mida i crida a la funció vital_to_zarr utilitzant 
    la configuració de freqüències.
    """

    if PRUEVAS:
        print("\n-- MODO PRUEVAS ACTIVADO (Streaming_to_zarr.py) ---")
    
    if not os.path.exists(vital_path):
        print(f"-- Error: El archivo {os.path.basename(vital_path)} ya no existe.")
        return -1, last_read_counts, True # Devuelve finished = True en caso de que el fitxero no exista
    
    current_size = os.path.getsize(vital_path)

    if not PRUEVAS and current_size == last_size: 
        return last_size, last_read_counts, False

    if not PRUEVAS and (last_size == -1 or current_size < last_size): 
        return current_size, last_read_counts, False
    
    print(f"\n-- El archivo {os.path.basename(vital_path)} se ha cambiado/simulado. Tamaño: {last_size if last_size != -1 else 0} -> {current_size} bytes.")

    new_last_read_counts = last_read_counts.copy()
    
    for attemp in range(3):
        try:
            print(f"\n-- INICIANT PROCESSAMENT ZARR AL FITXER: {STORE_PATH} (Streaming_to_zarr.py) ---")
            
            window_to_process = simulated_growth_seconds if PRUEVAS else None # En caso de no ser PRUEVAS, esto no sirve de nada

            new_last_read_counts = vital_to_zarr_streaming( 
                vital_path=vital_path,
                available_tracks=available_list,
                zarr_path=STORE_PATH, 
                last_read_counts=last_read_counts, 
                simulated_growth_seconds=window_to_process # Passarle al vital_to_zarr los segundos de simulacion
            )
            break # En caso que se llegue aqui sales del bucle, significado que se ha actualizado el fitxero

        except Exception as e:
            if attemp < 2:
                print(f"-- [WARN] Error al processar a Zarr (Intent {attemp+1}/3): {type(e).__name__}: {e}. Reintentant en 0.5 segons...")
                time.sleep(0.5)
                continue
            print(f"-- Error CRÍTIC al processar a Zarr: {type(e).__name__}: {e}")
            return last_size, last_read_counts, False # En caso de error critico, devolver el conteo

    if PRUEVAS:
        total_simulated_time = simulated_growth_seconds
        print(f"-- SIMULACIÓ: S'han processat {total_simulated_time} segons de dades noves. ---")
        # En modo PRUEVAS, solo acaba cuando llega al final de todo
        return current_size, new_last_read_counts, False 

    return current_size, new_last_read_counts, False # Devuelve los nuevos conteos


# --------------------------------------------------------------------------------------

def main_loop(stop_event: threading.Event, algoritmos_cargados_event: threading.Event, algoritmos_disponibles: list, vital_path, directorio_dia):

    session_timestamp = obtener_vital_timestamp(vital_path)

    print(f"-- Carpeta del día: {directorio_dia}")
    print(f"-- Archivo .vital más reciente: {os.path.basename(vital_path)}")
    print(f"-- Timestamp de Sesión (per a Zarr): {session_timestamp}")
    print(f"-- Directorio de salida ZARR (Acumulativo): {STORE_PATH}")
    print(f"-- Iniciando Polling cada {POLLING_INTERVAL} segundos")

    vf = vitaldb.VitalFile(vital_path)

    limpios = []
    for track in vf.get_track_names():
        vd = vf.to_pandas(track_names = track, interval = 0, return_timestamp=True)
        vd_clean = vd[vd[track].notna()]
        if track != "Intellivue/ALARM_STATUS" and track != "Intellivue/ALARM_PRIORITY" and track != "Intellivue/ALARM_MESSAGE":
            vd_clean2 = vd_clean[vd_clean[track] > -5.0]
            
            if track != "Intellivue/ICP":
                vd_clean3 = vd_clean2[vd_clean2[track] != 0.0]
            else: 
                vd_clean3 = vd_clean2[vd_clean2[track] > 0.0]
            
            if vd_clean3 is not []:
                if not vd_clean3.empty:
                    limpios.append(track)
    
    lista_algoritmos = check_availability(limpios)
    print("Algoritmos seleccionables: ", lista_algoritmos)
    algoritmos_disponibles_visibles = [alg for alg in lista_algoritmos if alg in VISIBLE_ALGORITHMS]
    print("Algoritmos visibles: ", algoritmos_disponibles_visibles)
    algoritmos_disponibles.extend(algoritmos_disponibles_visibles)

    algoritmos_cargados_event.set()

    last_size = -1
    last_read_counts = {} # Inicializacion de la variable

    if PRUEVAS:
        total_sim_cycles = 1000
        current_sim_cycle = 0

    try:
        while not stop_event.is_set():
            if PRUEVAS:
                simulated_growth_seconds = random.randint(SIM_MIN_SECS, SIM_MAX_SECS)
                print(f"\n-- SIMULACIÓN --: Leyendo bloque de {simulated_growth_seconds} segundos.")
            else:
                simulated_growth_seconds = 0

            current_size, last_read_counts, finished = verificar_y_procesar(
                vital_path, 
                limpios,
                last_size, 
                last_read_counts,
                simulated_growth_seconds
            )
            last_size = current_size 
            
            # Control de finalitzación de simulación
            if PRUEVAS:
                if finished:
                    print("\n-- SIMULACIÓN FINALIZADA: Archivo no disponible o terminado. ---")
                    stop_event.set()
                    break

                current_sim_cycle += 1

                if current_sim_cycle >= total_sim_cycles:
                    print(f"\n-- SIMULACIÓN FINALIZADA: {total_sim_cycles} ciclos completados. ---")
                    break
            
            time.sleep(POLLING_INTERVAL)

            #time.sleep(simulated_growth_seconds)

    except KeyboardInterrupt:
        print("\n-- Finalizando Polling.")

if __name__ == "__main__":
    print("-- Prueva del Streaming en Thread")
    stop_event = threading.Event()
    main_loop(stop_event)
