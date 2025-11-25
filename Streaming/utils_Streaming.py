import os
import re
import pandas as pd
import datetime

OUTPUT_DIR = r"C:\Users\UX636EU\OneDrive - EY\Desktop\VitalParser-main\VitalParser-main\results" # Directorio donde se guardarán los archivos CSV de onda

#os.makedirs(OUTPUT_DIR, exist_ok=True) # En caso de que el directorio no exista, se crea


WAVE_STANDARD_RATE = 100.0 # Frecuencia de muestreo estándar asumida para tracks no definidos (en Hz)

# Si un track está aquí, usa su valor. Si no está, usa WAVE_STANDARD_RATE (100.0 Hz).
WAVE_TRACKS_FREQUENCIES = {
    'Intellivue/ABP': 125.0,    # Frecuencia personalizada en caso de saber qual es, en Hz
    'Intellivue/AOP': 125.0,  
    'Intellivue/ART': 125.0,    
    'Intellivue/AWP': 62.5,        # Si se pone 0, usará WAVE_STANDARD_RATE
    'Intellivue/CO2': 62.5,
    'Intellivue/CVP': 125.0,
    'Intellivue/ECG_AI': 0.0, # Ni idea de este
    'Intellivue/ECG_AS': 0.0, # Ni idea de este
    'Intellivue/ECG_AVR': 500.0,
    'Intellivue/ECG_ES': 0.0, # Ni idea de este
    'Intellivue/ECG_I': 500.0,
    'Intellivue/ECG_II': 500.0,
    'Intellivue/ECG_III': 500.0,
    'Intellivue/ECG_V': 500.0,
    'Intellivue/EEG': 125.0,
    'Intellivue/FLOW': 62.5, ## Ni idea de este
    'Intellivue/ICP': 125.0,
    'Intellivue/PLETH': 125.0,
    'Intellivue/RESP': 62.5,
    'Demo/ECG': 100.0,
    'Demo/PLETH': 100.0,
    'Demo/ART': 100.0,
    'Demo/CVP': 100.0,
    'Demo/EEG': 100.0,
    'Demo/CO2': 100.0,
}

# -------------------------------------------------------------------------------------------

def obtener_vital_timestamp(vital_path): 
    """
    Extrae la parte YMD_HMS (ej. '251025_154310') del nombre del archivo .vital.
    En formato PRUEVAS no se usa
    """
    filename = os.path.basename(vital_path)
    match = re.search(r'(\d{6}_\d{6})\.vital$', filename) # Busca el patrón de 6 dígitos + _ + 6 dígitos antes de .vital
    if match:
        return match.group(1)
    return "UnknownTimestamp"

def guardar_muestras_csv(track_name, samples_array, start_index, session_timestamp, time_offsets_array):
    """
    Exporta el array de muestras (WAVE o NUM) con su timestamp a un archivo CSV.
    Usa el session_timestamp en el nombre del archivo.
    Si el time_offsets_array es None, genera timestamps basados en el start_index y real_rate (para NUM).
    Agrega los nuevos datos al final del archivo CSV con nombre fijo para la sesión/track.
    """
    if len(samples_array) == 0: # No hay muestras para guardar
        return

    try:
        # Crear DataFrame con timestamps y valores
        df = pd.DataFrame({
        'Time (s)': time_offsets_array,
        'Value': samples_array,
        })
        
        # Determinar el nombre y la ruta del archivo (NOMBRE FIJO con TIMESTAMP de sesión)
        safe_track_name = track_name.replace('/', '_').replace(' ', '_')
        
        # TrackName_VARIABLE_TIMESTAMP.csv (ej. Intellivue_ECG_WAVE_251025_154310.csv, Intellivue_ABP_NUM_251025_154310.csv)
        label = "_WAVE_" if track_name in WAVE_TRACKS_FREQUENCIES else "_NUM_"
        filename = f"{safe_track_name}{label}_{session_timestamp}.csv" 
        filepath = os.path.join(OUTPUT_DIR, filename)

        # Guardar en CSV en modo de 'append' (a)
        header_needed = not os.path.exists(filepath)
        df.to_csv(filepath, mode='a', header=header_needed, index=False)
        print(f"  > **AÑADIDO:** {len(samples_array)} puntos al archivo '{filename}' (Total hasta ahora: {start_index + len(samples_array)} puntos)")

    except Exception as e:
        print(f"  > Error al guardar el CSV para {track_name}: {e}")
        
def obtener_directorio_del_dia(base_dir):
    """
    Obtiene el directorio basandose en el dia de hoy
    En el formato PRUEVAS no se usa
    """
    directorio_dia = os.path.join(base_dir, datetime.datetime.now().strftime("%y%m%d"))
    if not os.path.exists(directorio_dia):
        raise FileNotFoundError(f"No existe la carpeta para hoy: {directorio_dia}")
    return directorio_dia

def obtener_vital_mas_reciente(directorio):
    """
    Busca en el directorio el archivo .vital con el timestamp mas grande (Por lo tanto el mas viejo o el ultimo creado)
    En el formato PRUEVAS no se usa
    """
    archivos = [f for f in os.listdir(directorio) if f.endswith(".vital")]
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos .vital en {directorio}")

    with_timestamp = []

    for fname in archivos:
        m = re.compile(r'_(\d{6})_(\d{6})\.vital$').search(fname)
        if not m:
            continue
        fullpath = os.path.join(directorio, fname)
        yymmdd = m.group(1)
        hhmmss = m.group(2)
        
        yy = int(yymmdd[0:2])
        mm = int(yymmdd[2:4])
        dd = int(yymmdd[4:6])
        hh = int(hhmmss[0:2])
        mi = int(hhmmss[2:4])
        ss = int(hhmmss[4:6])
        dt = datetime.datetime(2000 + yy, mm, dd, hh, mi, ss)
        with_timestamp.append((fullpath, dt))

    if not with_timestamp:
        raise FileNotFoundError(f"No se encontraron archivos .vital válidos en {directorio}")

    with_timestamp.sort(key=lambda x: x[1], reverse=True)
    return with_timestamp[0][0]
