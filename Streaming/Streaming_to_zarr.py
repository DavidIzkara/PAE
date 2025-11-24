import os
import time
import numpy as np
import random 
from utils_zarr import safe_group, get_group_if_exists, append_1d, open_root
from utils_Streaming import OUTPUT_DIR, WAVE_TRACKS_FREQUENCIES, WAVE_STANDARD_RATE, obtener_vital_timestamp, obtener_directorio_del_dia, obtener_vital_mas_reciente
from numcodecs import Blosc 
from vitaldb import VitalFile 

BASE_DIR = r"C:\Users\UX636EU\OneDrive - EY\Desktop\recordings" 
POLLING_INTERVAL = 1 

# -------------------------------------------------------------------------------------------
PRUEVAS = True 

DIRECTORIO_PRUEVA = r"C:\Users\UX636EU\OneDrive - EY\Desktop\VitalParser-main\VitalParser-main\records\4\250618" 
ARCHIVO_VITAL = r"nd7wx5v9h_250618_133910.vital" 

SIM_MIN_SECS = 20
SIM_MAX_SECS = 30

# -------------------------------------------------------------------------------------------
# CONFIGURACIÓ ZARR
# -------------------------------------------------------------------------------------------
ZARR_PATH = os.path.join(OUTPUT_DIR, "session_data.zarr") 
_COMPRESSOR = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)


def vital_to_zarr_streaming(
    vital_path: str,
    zarr_path: str,
    last_read_counts: dict, 
    simulated_growth_seconds: float | None = None, # Renombrado de window_secs
    chunk_len: int = 30000,
) -> dict: 
    """
    Processa l'arxiu vital utilitzant la lògica de freqüències i el slicing per 
    conteig de mostres (last_read_counts) en mode simulació.
    """
    if not os.path.exists(vital_path):
        raise FileNotFoundError(f"No s'ha trobat el .vital: {vital_path}")

    vf = VitalFile(vital_path)
    available_tracks = vf.get_track_names() 

    #os.makedirs(os.path.dirname(zarr_path) or ".", exist_ok=True)
    root = open_root(zarr_path)

    signals_root = safe_group(root, "signals")

    written_any = False
    total_added_samples = 0
    written_tracks = 0
    skipped_no_new = 0
    skipped_empty = 0
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Preparam el nou diccionari de conteos basat en l'anterior
    new_last_read_counts = last_read_counts.copy() 

    for track in available_tracks:
        rate = 0.5 
        interval = 1.0 
        track_type = "NUM"
        
        # Determinació de la freqüència i l'interval
        if track in WAVE_TRACKS_FREQUENCIES:
            track_type = "WAVE"
            rate_configurada = WAVE_TRACKS_FREQUENCIES.get(track, 0)
            if rate_configurada > 0:
                rate = rate_configurada
            else:
                rate = WAVE_STANDARD_RATE 
            interval = 1.0 / rate 

        # Llegir TOTA la data disponible des del .vital
        try:
            data = vf.to_numpy([track], interval=interval, return_timestamp=True)
        except Exception as e:
            print(f"[WARN] No s'ha pogut llegir el track '{track}' amb interval={interval}: {e}")
            skipped_empty += 1
            continue

        if data is None or data.size == 0:
            skipped_empty += 1
            continue

        ts = data[:, 0].astype(np.float64, copy=False)

        raw_vals = data[:, 1]
        vals = np.full(raw_vals.size, np.nan, dtype=np.float64)

        for i, val in enumerate(raw_vals):
            try:
                vals[i] = float(val)
            except (TypeError, ValueError):
                vals[i] = np.nan

        current_total_count = len(vals)
        last_count = last_read_counts.get(track, 0)
        end_index = current_total_count # Per defecte, llegir tot (MODO REAL)
        
        # --- LÒGICA DE SLICING PER CONTEO (Simulació) ---
        if simulated_growth_seconds is not None and simulated_growth_seconds > 0: 
            
            simulated_added_samples = int(simulated_growth_seconds * rate) 
            target_count = min(last_count + simulated_added_samples, current_total_count) 
            
            if target_count <= last_count: 
                skipped_no_new += 1
                new_last_read_counts[track] = last_count # Mantenir el comptador antic
                continue
                
            end_index = target_count # El nou índex final per al slicing.

        # Aplicar el tall de mostres (des de last_count fins a end_index)
        ts_slice = ts[last_count:end_index] 
        vals_slice = vals[last_count:end_index]
        
        if ts_slice.size == 0:
            skipped_no_new += 1
            new_last_read_counts[track] = last_count # Mantenir el comptador antic
            continue

        # Actualitzar el conteo de mostres per al pròxim cicle
        new_last_read_counts[track] = end_index 
        
        # Convertir a ms i float32
        ts_ms = np.rint(ts_slice * 1000.0).astype(np.int64)
        vals_f32 = vals_slice.astype(np.float32)

        
        # --- Deduplicació Zarr (Chequeo de seguretat per timestamp) ---
        # Aquest pas garanteix que no s'escrigui data duplicada al Zarr 
        # si hi hagués una inconsistència entre last_read_counts i el Zarr final.
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
                print(f"[WARN] No s'ha pogut llegir l'últim time_ms de '{track}' per a deduplicació: {e}")
                pass
                
        # Escriure a Zarr
        grp = safe_group(signals_root, track)

        try:
            if "time_ms" in grp and "value" in grp:
                ds_time = grp["time_ms"]
                ds_val = grp["value"]
            else:
                ds_time = grp.require_dataset("time_ms", shape=(0,), maxshape=(None,), chunks=(chunk_len,), dtype="int64", compressor=_COMPRESSOR)
                ds_val = grp.require_dataset("value", shape=(0,), maxshape=(None,), chunks=(chunk_len,), dtype="float32", compressor=_COMPRESSOR)
        except Exception as e:
            raise Exception(f"Falla la creació o el resize de l'array Zarr per a '{track}': {type(e).__name__}: {e}")

        append_1d(ds_time, ts_ms)
        append_1d(ds_val, vals_f32)

        grp.attrs["track"] = track
        grp.attrs["sampling_rate_hz"] = rate
        grp.attrs["track_type"] = track_type 
        grp.attrs["last_updated"] = timestamp
        grp.attrs["total_samples"] = ds_time.shape[0]
        grp.attrs["duration_seconds"] = ds_time.shape[0] / rate if rate > 0 else 0.0
        if PRUEVAS:
            grp.attrs["last_update_secs"] = simulated_growth_seconds if simulated_growth_seconds is not None else 0.0
        else:
            if (current_total_count - last_count) / rate < round((current_total_count - last_count) / rate):
                grp.attrs["last_update_secs"] = (round((current_total_count - last_count) / rate)) - 1
            else: # En caso que el calcul dongui un valor menor al real
                grp.attrs["last_update_secs"] = round((current_total_count - last_count) / rate)
        grp.attrs["last_update_samples"] = ts_ms.size
        
        print(f"[{track}: {track_type} {rate:.1f} Hz] +{ts_ms.size} mostres (total={ds_time.shape[0]})")
        total_added_samples += int(ts_ms.size)
        written_tracks += 1
        written_any = True

    root.attrs.setdefault("schema", "v1")
    root.attrs.setdefault("created_by", "Streaming")
    root.attrs.setdefault("time_origin", "epoch1700_ms")
    root.attrs["last_updated"] = timestamp

    # Missatge final de resum
    if written_any:
        print(f"\n✅ Escrita/actualitzada la col·lecció a: {zarr_path}")
        print(f"   Tracks actualitzades: {written_tracks}, mostres afegides: {total_added_samples}")
    else:
        print(f"\n⚠️  No s'ha escrit cap mostra nova.")
        
    return new_last_read_counts # <-- Retornem els nous conteos


# --- MODIFICACIÓ DE verificar_y_procesar ---
def verificar_y_procesar(vital_path, last_size, last_read_counts, simulated_growth_seconds):
    """
    Comprova la mida i crida a la funció vital_to_zarr utilitzant 
    la configuració de freqüències.
    """

    if PRUEVAS:
        print("\n--- MODO PRUEVAS ACTIVADO ---")
    
    if not os.path.exists(vital_path):
        print(f" Error: El archivo {os.path.basename(vital_path)} ya no existe.")
        return -1, last_read_counts, True # Retornem finished=True si el fitxer ha desaparegut
    
    current_size = os.path.getsize(vital_path)

    if not PRUEVAS and current_size == last_size: 
        return last_size, last_read_counts, False

    if not PRUEVAS and (last_size == -1 or current_size < last_size): 
        return current_size, last_read_counts, False
    
    print(f"\n El archivo {os.path.basename(vital_path)} se ha cambiado/simulado. Tamaño: {last_size if last_size != -1 else 0} -> {current_size} bytes.")

    new_last_read_counts = last_read_counts.copy()
    
    # --- BLOC DE PROCESSAMENT ZARR AMB FREQÜÈNCIES ---
    for attemp in range(3):
        try:
            print(f"\n--- INICIANT PROCESSAMENT ZARR AL FITXER: {ZARR_PATH} ---")
            
            window_to_process = simulated_growth_seconds if PRUEVAS else None

            # CRIDA MODIFICADA: Passem i capturem el diccionari de conteos
            new_last_read_counts = vital_to_zarr_streaming( 
                vital_path=vital_path,
                zarr_path=ZARR_PATH, 
                last_read_counts=last_read_counts, 
                simulated_growth_seconds=window_to_process # Passar els segons simulats
            )
            break # Sortir del bucle de reintent en cas d'èxit

        except Exception as e:
            if attemp < 2:
                print(f" [WARN] Error al processar a Zarr (Intent {attemp+1}/3): {type(e).__name__}: {e}. Reintentant en 0.5 segons...")
                time.sleep(0.5)
                continue
            print(f" Error CRÍTIC al processar a Zarr: {type(e).__name__}: {e}")
            return last_size, last_read_counts, False # Retornem els conteos antics en cas d'error crític

    if PRUEVAS:
        total_simulated_time = simulated_growth_seconds
        print(f"--- SIMULACIÓ: S'han processat {total_simulated_time} segons de dades noves. ---")
        # En mode simulació, assumim que no ha acabat fins que el bucle principal ho determini.
        return current_size, new_last_read_counts, False 

    return current_size, new_last_read_counts, False # <--- Retornem els nous conteos


# --------------------------------------------------------------------------------------

# --- MODIFICACIÓ DE main_loop ---
def main_loop():
    if PRUEVAS:
        directorio_dia = DIRECTORIO_PRUEVA
        vital_path = os.path.join(DIRECTORIO_PRUEVA, ARCHIVO_VITAL)

        print(f" SIMULACIÓN DE ACTUALIZACIÓN: {SIM_MIN_SECS}-{SIM_MAX_SECS} segundos por ciclo.")
        print(f" Iniciando Polling cada {POLLING_INTERVAL} segundos")
    # ... (mode real es manté) ...
    else:
        try:
            directorio_dia = obtener_directorio_del_dia(BASE_DIR)
            vital_path = obtener_vital_mas_reciente(directorio_dia)
        except FileNotFoundError as e:
            print(f" Error: {e}")
            return

    session_timestamp = obtener_vital_timestamp(vital_path)

    print(f" Carpeta del día: {directorio_dia}")
    print(f" Archivo .vital más reciente: {os.path.basename(vital_path)}")
    print(f" Timestamp de Sesión (per a Zarr): {session_timestamp}")
    print(f" Directorio de salida ZARR (Acumulativo): {ZARR_PATH}")
    print(f" Iniciando Polling cada {POLLING_INTERVAL} segundos")

    last_size = -1
    last_read_counts = {} # <--- INICIALITZAT EL DICCIONARI

    try:
        while True:
            if PRUEVAS:
                simulated_growth_seconds = random.randint(SIM_MIN_SECS, SIM_MAX_SECS)
                print(f"\n--- SIMULACIÓN ---: Leyendo bloque de {simulated_growth_seconds} segundos.")
            else:
                simulated_growth_seconds = 0

            # CRIDA MODIFICADA: Capturem la mida, el diccionari de conteos i l'estat 'finished'
            current_size, last_read_counts, finished = verificar_y_procesar(
                vital_path, 
                last_size, 
                last_read_counts, # <-- Passar el diccionari
                simulated_growth_seconds
            )
            last_size = current_size 
            
            # Control de finalització de simulació
            if PRUEVAS:
                if finished:
                    print("\n--- SIMULACIÓN FINALIZADA: Archivo no disponible o terminado. ---")
                    break
            
            time.sleep(POLLING_INTERVAL)

    except KeyboardInterrupt:
        print("\n Finalizando Polling.")

if __name__ == "__main__":
    main_loop()
