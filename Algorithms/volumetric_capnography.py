import vitaldb
import numpy as np
import pandas as pd

class VolumetricCapnography:

    def __init__(self):
        self.fs= 100  # Sampling frequency
        self.last_flow = np.array([])
        self.last_co2 = np.array([])
        self.last_time = np.array([])
        self.values = None
    
    def compute(self,data):
        if isinstance(data, vitaldb.VitalFile):
            self._from_vf(data)
        else:
            self._from_df(data)
        return self.values
    
    def _from_vf(self, vf):
        # Get all available track names in the VitalFile
        available_tracks = vf.get_track_names()
        
        co2_track = next(
            (t for t in available_tracks if 'Intellivue/CO2' in t),
            next((t for t in available_tracks if 'Intellivue/AWAY_CO2_ET' in t),
                next((t for t in available_tracks if 'Intellivue/AWAY_CO2_INSP_MIN' in t),None)))
        
        flow_track = next(
            (t for t in available_tracks if 'Intellivue/FLOW' in t),
            next((t for t in available_tracks if 'Intellivue/RESP' in t),
                next((t for t in available_tracks if 'Intellivue/TV_EXP' in t),None)))

        co2 = vf.to_pandas(track_names=co2_track, interval=1/62.5, return_timestamp=True)
        flow = vf.to_pandas(track_names=flow_track, interval=1/62.5, return_timestamp=True)
        
        df = self.compute_sync(flow, co2)
        self.values = self.compute_volcap(df)

    def _from_df(self, list_dataframe:dict[str, pd.DataFrame]):
        available_tracks = list_dataframe.keys()

        co2_track = next(
            (t for t in available_tracks if 'Intellivue/CO2' in t),
            next((t for t in available_tracks if 'Intellivue/AWAY_CO2_ET' in t),
                next((t for t in available_tracks if 'Intellivue/AWAY_CO2_INSP_MIN' in t),None)))
        
        assert co2_track is not None
        co2_raw = list_dataframe[co2_track]
        co2= pd.DataFrame({co2_track:co2_raw["value"], 'Time': co2_raw["time_ms"] })

        flow_track = next(
            (t for t in available_tracks if 'Intellivue/FLOW' in t),
            next((t for t in available_tracks if 'Intellivue/RESP' in t),
                next((t for t in available_tracks if 'Intellivue/TV_EXP' in t),None)))
        
        assert flow_track is not None
        flow_raw = list_dataframe[flow_track]
        flow= pd.DataFrame({flow_track:flow_raw["value"], 'Time': flow_raw["time_ms"] })

        df = self.compute_sync(flow, co2)
        self.values = self.compute_volcap(df)
    
    #sincronizacion
    def compute_sync(self,flow_df,co2_df):
        
        t_flow= flow_df['Time'].values
        t_co2= co2_df['Time'].values
        co2_interp=np.interp(t_flow,t_co2,co2_df['value'].values)
        return pd.DataFrame({'Time':t_flow,'Flow':flow_df['value'].values,'CO2':co2_interp})

    def compute_exh(self,flow,min=20,thr=0.05):
        is_exp= flow>thr #thr= 0.05 L/s
        diff= np.diff(is_exp.astype(int))
        starts= np.where(diff==1)[0]+1 # bordes ↑ (inicio espiración) (si es posición 2, la muestra de inicio es la 3)
        ends= np.where(diff==-1)[0]+1  # bordes ↓ (fin espiración) (si es posición 2, la muestra de fin es la 3)

        if (len(ends)==0 and len(starts)==0 ) or ends[0]<starts[0]:
            ends= ends[1:] # eliminar fin sin inicio
        n= min(len(starts),len(ends))
        starts= starts[:n]
        ends= ends[:n]

        #filtrar por duración mínima
        valid_mask = [(e - s) >= min for s, e in zip(starts, ends)] 
        """
        Ejemplo:
        starts = [125, 350, 575, 800]      # índices de inicio de espiraciones
        ends   = [200, 470, 695, 820]      # índices de fin de espiraciones
        min_dur_samples = 100              # duración mínima = 100 muestras (1s a 100Hz)

        zip->
        (125, 200), (350, 470), (575, 695), (800, 820)
        s=125, e=200 → duración = 200-125 = 75 muestras
        s=350, e=470 → duración = 470-350 = 120 muestras  
        s=575, e=695 → duración = 695-575 = 120 muestras
        s=800, e=820 → duración = 820-800 = 20 muestras

        75 >= 100 → False  (demasiado corta)
        120 >= 100 → True  (válida)
        120 >= 100 → True  (válida)
        20 >= 100 → False  (demasiado corta)
        valid_mask = [False, True, True, False]
        """
        starts_valid = np.array(starts)[valid_mask]
        ends_valid = np.array(ends)[valid_mask]
        durations = ends_valid - starts_valid
        return  pd.DataFrame({'idx_ini':starts_valid,'idx_fin':ends_valid,'duration_samples':durations})
    
    def compute_volcap(self, df):
        
        t=df['Time'].values
        flow=df['Flow'].values
        co2=df['CO2'].values
        dt= 1/self.fs # intervalo de muestreo (s)
        vol_total= np.cumsum(flow*dt)  # volumen total (L)
        exh_df= self.compute_exh(flow)
        resultados=[]

        for _,row in exh_df.iterrows():
            # junta el indice con cada fila, el indice no nos interesa
            ini= int(row['idx_ini'])
            fin= int(row['idx_fin'])
            duration= row['duration_samples']*dt  # duración en segundos
            idx= np.arange(ini,fin) #si ini=125, fin=245,idx[125,126,...244]
        
            vol_rel= vol_total[idx]- vol_total[ini]  # volumen espirado relativo 
            co2_exh= co2[idx]  # valores de CO2 durante la espiración
            vt= vol_rel[-1]  # volumen tidal espirado
            vco2= np.trapz(co2_exh,vol_rel) # area bajo la curva volumen-CO2
            resultados.append([t[ini], t[fin],duration, vt, vco2])

        return pd.DataFrame(resultados, columns=['time_ini_ms','time_fin_ms','duration_s','VT','VCO2'])
