import vitaldb
import numpy as np
from scipy.stats import linregress
from scipy.signal import find_peaks
from compute_rr import compute_rr

class BaroreflexSensitivity:
    
    def __init__(self,data):
        
        if isinstance(data, vitaldb.VitalFile):
            self._from_vf(data)
        else:
            self._from_df(data)
    def _from_vf(self, vf):
        # Get all available track names in the VitalFile
        available_tracks = vf.get_track_names()

        # Try to find heart rate wave
        hr_track = next(
            (t for t in available_tracks if 'Intellivue/ECG_I' in t), 
            next((t for t in available_tracks if 'Intellivue/ECG_II' in t),     
                 next((t for t in available_tracks if 'Intellivue/ECG_III' in t), 
                      next((t for t in available_tracks if 'Intellivue/ECG_V' in t), None)))) 

        # Try to find arterial pressure wave
        art_track = next(
            (t for t in available_tracks if 'Intellivue/ART' in t), 
            next((t for t in available_tracks if 'Intellivue/ABP' in t), None))


        hr = vf.to_pandas(track_names=hr_track, interval=1/500, return_timestamp=True)
        rr = compute_rr(hr, hr_track)


        art_raw = vf.to_pandas(track_names=art_track, interval=1/500, return_timestamp=True)
        art = pd.DataFrame({'value': art_raw[art_track], 'Time': art_raw.index})
        sbp = self.compute_sbp(art)

        brs = self.compute_brs(sbp, rr)
        
        values = brs
        
        self.values = pd.DataFrame({'Time_ini_ms': values["Time_ini_ms"], 'Time_fin_ms': values["Time_fin_ms"], 'BRS': values["BRS"]})

    def _from_df(self, list_dataframe: list[pd.DataFrame]):
        available_tracks = list_dataframe.keys()

        # Try to find heart rate wave
        hr_track = next(
            (t for t in available_tracks if 'Intellivue/ECG_I' in t), 
            next((t for t in available_tracks if 'Intellivue/ECG_II' in t),     
                 next((t for t in available_tracks if 'Intellivue/ECG_III' in t), 
                      next((t for t in available_tracks if 'Intellivue/ECG_V' in t), None)))) 
        
        hr_raw = list_dataframe[hr_track]
        hr = pd.DataFrame({hr_track:hr_raw["value"], 'Time': hr_raw["time_ms"] })
        rr = compute_rr(hr, hr_track)

        # Try to find arterial pressure wave
        art_track = next(
            (t for t in available_tracks if 'Intellivue/ART' in t), 
            next((t for t in available_tracks if 'Intellivue/ABP' in t), None))

        art_raw = list_dataframe[art_track]
        art = pd.DataFrame({'value': art_raw["value"], 'Time': art_raw["time_ms"]})
        sbp = self.compute_sbp(art)

        # Calculate BRS metrics
        brs = self.compute_brs(sbp, rr)
        
        values = brs

        self.values = pd.DataFrame({'Time_ini_ms': values["Time_ini_ms"], 'Time_fin_ms': values["Time_fin_ms"], 'BRS': values["BRS"]})

    def compute_sbp(self, art_signal):
        """
        Calcula la Presión Sistólica y determina el inicio y fin del ciclo de presión.
        Time_ini: Momento de la diástole previa (valle).
        Time_fin: Momento de la diástole siguiente (valle).
        """
        fs=500  # Frecuencia de muestreo en Hz
        signal = art_signal['value'].values
        times = art_signal['Time'].values

        # 1. Detectar Picos Sistólicos (Máximos)
        # distance=100 (0.2s) para evitar ruido
        peaks_idx, _ = find_peaks(signal, distance=int(fs*0.25), height=40)
        
        # 2. Detectar Valles Diastólicos (Mínimos)
        # Invertimos la señal para encontrar los mínimos usando find_peaks
        valleys_idx, _ = find_peaks(-signal, distance=int(fs*0.25))

        results = []

        # Usamos searchsorted para encontrar los valles que rodean a cada pico
        # Esto es mucho más eficiente que iterar manualmente
        if len(valleys_idx) > 1 and len(peaks_idx) > 0:
            # Para cada pico, buscamos dónde encaja en la lista de valles
            insert_positions = np.searchsorted(valleys_idx, peaks_idx)
            
            for i, p_idx in enumerate(peaks_idx):
                pos = insert_positions[i]
                
                # Validamos que el pico tenga un valle antes y un valle después
                if pos > 0 and pos < len(valleys_idx):
                    valley_prev_idx = valleys_idx[pos - 1] # Inicio del ciclo
                    valley_next_idx = valleys_idx[pos]     # Fin del ciclo
                    
                    results.append([
                        times[valley_prev_idx], # Time_ini_ms
                        times[valley_next_idx], # Time_fin_ms
                        signal[p_idx]           # Valor SBP
                    ])

        return pd.DataFrame(results, columns=['Time_ini_ms', 'Time_fin_ms', 'sbp'])

        
     
    def compute_brs(self, sbp_df, rr_df):
        # Extraer arrays para velocidad
        rr = rr_df['rr'].values
        ts_ini = rr_df['Time_ini_ms'].values # Tiempos del ECG
        ts_fin = rr_df['Time_fin_ms'].values
        
        sbp = sbp_df['sbp'].values
        # sbp_df también tiene Time_ini_ms/Time_fin_ms si se necesitan para debugging,
        # pero para el resultado final usaremos los tiempos alineados de la secuencia.

        brs_results = []
        
        # Iteramos buscando secuencias de 3 latidos
        n = min(len(sbp), len(rr)) - 2
        
        for i in range(n):
            # Secuencia de SUBIDA: SBP sube y RR sube (retraso fisiológico normal)
            is_up = (sbp[i] < sbp[i+1] < sbp[i+2]) and (rr[i] < rr[i+1] < rr[i+2])
            
            # Secuencia de BAJADA: SBP baja y RR baja
            is_down = (sbp[i] > sbp[i+1] > sbp[i+2]) and (rr[i] > rr[i+1] > rr[i+2])

            if is_up or is_down:
                # Regresión lineal entre presión (x) y intervalo RR (y)
                slope, _, r_value, _, _ = linregress(sbp[i:i+3], rr[i:i+3])
                
                # Umbral de correlación
                if r_value > 0.6: 
                    # Definimos el tiempo de la secuencia:
                    # Inicio: Comienzo del primer latido de la secuencia
                    t_start = ts_ini[i]
                    # Fin: Final del último latido de la secuencia
                    t_end = ts_fin[i+2]
                    
                    brs_results.append([t_start, t_end, slope])

        # Devolver DataFrame o vacío si no hay resultados
        if len(brs_results) > 0:
            return pd.DataFrame(brs_results, columns=["Time_ini_ms", "Time_fin_ms", "BRS"])
        else:
            print("No se encontraron secuencias válidas para BRS.")
            return pd.DataFrame(columns=["Time_ini_ms", "Time_fin_ms", "BRS"])
