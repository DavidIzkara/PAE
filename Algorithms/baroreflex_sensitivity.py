import vitaldb
import numpy as np
from scipy.stats import linregress
from scipy.signal import find_peaks
from compute_rr import compute_rr
import pandas as pd

class BaroreflexSensitivity:
    
    def __init__(self,data):
        
        self.last2_rr = []
        self.last2_sbp = []
        self.last2_ini = []
        self.last2_fin = []
        self.values = None

    def compute(self, data):
        if isinstance(data, vitaldb.VitalFile):
            self._from_vf(data)
        else:
            self._from_df(data)
        return self.values


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

        self.values = self.compute_brs(sbp, rr)
        
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

        self.values = self.compute_brs(sbp, rr)

    def compute_sbp(self, art_signal):
        """
        Calcula la Presión Sistólica y determina el inicio y fin del ciclo de presión.
        Time_ini: Momento de la diástole previa (valle).
        Time_fin: Momento de la diástole siguiente (valle).
        """
        fs=500  # Frecuencia de muestreo en Hz
        signal = art_signal['value'].values
        times = art_signal['Time'].values

        # Detectar Picos Sistólicos (Máximos)
        # distance=100 (0.2s) para evitar ruido
        peaks_idx, _ = find_peaks(signal, distance=int(fs*0.25), height=40)
        
        # Detectar Valles Diastólicos (Mínimos)
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
        # Extraer arrays nuevos
        rr_new = rr_df['rr'].values.tolist()
        ts_ini_new = rr_df['Time_ini_ms'].values.tolist()
        ts_fin_new = rr_df['Time_fin_ms'].values.tolist()
        sbp_new = sbp_df['sbp'].values.tolist()

        # Alinear longitud de los nuevos datos (como hacías con min(len))
        min_len = min(len(sbp_new), len(rr_new))
        rr_new = rr_new[:min_len]
        sbp_new = sbp_new[:min_len]
        ts_ini_new = ts_ini_new[:min_len]
        ts_fin_new = ts_fin_new[:min_len]

        # Concatenar con el buffer anterior
        rr = self.last2_rr + rr_new
        sbp = self.last2_sbp + sbp_new
        ts_ini = self.last2_ini + ts_ini_new
        ts_fin = self.last2_fin + ts_fin_new

        brs_results = []
    
        n = len(rr) - 2 
        
        if n > 0:
            for i in range(n):
                # Secuencia de SUBIDA
                is_up = (sbp[i] < sbp[i+1] < sbp[i+2]) and (rr[i] < rr[i+1] < rr[i+2])
                
                # Secuencia de BAJADA
                is_down = (sbp[i] > sbp[i+1] > sbp[i+2]) and (rr[i] > rr[i+1] > rr[i+2])

                if is_up or is_down:
                    slope, _, r_value, _, _ = linregress(sbp[i:i+3], rr[i:i+3])
                    
                    if r_value > 0.6: 
                        t_start = ts_ini[i]
                        t_end = ts_fin[i+2]
                        brs_results.append([t_start, t_end, slope])

            # Guardamos los últimos 2 para el siguiente ciclo
            self.last2_rr = rr[-2:]
            self.last2_sbp = sbp[-2:]
            self.last2_ini = ts_ini[-2:]
            self.last2_fin = ts_fin[-2:]
        
        else:
            # Si no hay suficientes datos, guardamos todo en el buffer y esperamos
            self.last2_rr = rr
            self.last2_sbp = sbp
            self.last2_ini = ts_ini
            self.last2_fin = ts_fin

        return pd.DataFrame(brs_results, columns=["Time_ini_ms", "Time_fin_ms", "BRS"])

