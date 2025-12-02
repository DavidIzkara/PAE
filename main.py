import time
from Algorithms import ejecutar_algoritmos
from Front import Interface
from Streaming.Streaming_to_zarr import main_loop
from Streaming.zarr_to_algorithms import main_to_loop
from Zarr.utils_zarr_corrected import ALGORITMOS_VISIBLES, STORE_PATH, escribir_prediccion


def seleccionar_modo_gui():
    try:
        import tkinter as tk
    except Exception:
        # Si tkinter no está disponible (o no hay display), volver al input por consola
        try:
            return input("Selecciona modo (online/offline): ")
        except Exception:
            return None

    selection = {'mode': None}

    root = tk.Tk()
    root.title('Seleccionar modo')
    root.resizable(False, False)

    # Centrar la ventana de forma simple
    width = 300
    height = 120
    try:
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        root.geometry(f"{width}x{height}+{x}+{y}")
    except Exception:
        root.geometry(f"{width}x{height}")

    def set_mode(m):
        selection['mode'] = m
        root.quit()

    lbl = tk.Label(root, text='Selecciona modo:')
    lbl.pack(pady=(12, 6))

    frm = tk.Frame(root)
    frm.pack(pady=(0, 8))

    btn_off = tk.Button(frm, text='Offline', width=10, command=lambda: set_mode('offline'))
    btn_off.pack(side='left', padx=10)

    btn_on = tk.Button(frm, text='Online', width=10, command=lambda: set_mode('online'))
    btn_on.pack(side='left', padx=10)

    # Si el usuario cierra la ventana, salir sin seleccionar
    root.protocol('WM_DELETE_WINDOW', lambda: set_mode(None))

    try:
        root.mainloop()
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    return selection['mode']

def iniciar_streaming_en_thread():
    
    results = main_to_loop() # Funcion principal del zarr_to_algorithms.py
    
    for NombreAlgoritmo, df_result in results.items():
        value_columns = [col for col in df_result.columns if col != 'Timestamp' and col != 'Time_ini_ms' and col != 'Time_fin_ms']
        if 'Timestamp' in df_result.columns:
            time_ms_array = df_result['Timestamp'].values
            time_count = 1
        else:
            time_ini_ms_array = df_result['Time_ini_ms'].values
            time_fin_ms_array = df_result['Time_fin_ms'].values
            time_count = 2
        
        visible = NombreAlgoritmo in ALGORITMOS_VISIBLES

        for track_name in value_columns:
            value_array = df_result[track_name].values
            if time_count == 1:
                escribir_prediccion(STORE_PATH, track_name, time_ms_array, value_array, model_info = {"model": NombreAlgoritmo, "visibilidad": visible})
            else:
                escribir_prediccion(STORE_PATH, track_name, time_ini_ms_array, value_array, model_info = {"model": NombreAlgoritmo, "visibilidad": visible}, timestamps_fin_ms=time_fin_ms_array)

    return results

def main():
    modo = seleccionar_modo_gui()

    if modo is None:
        print("No se seleccionó modo. Saliendo.")
        return

    try:
        if modo == "offline":
            print("Modo offline activado. Esperando acciones...")
            #recordings_dir = r"" pendiente poner el path del vital_file
            vital_path = find_latest_vital(recordings_dir)
            vf = VitalFile(vital_path)
            
            tracks = vf.get_track_names()
            possible_list = check_availability(tracks)
            print("Los algoritmos disponibles son: ", possible_list )
            
            results = {}
            for algorithm in possible_list:
            
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
                if algorithm == 'Heart Rate Variability':
                    results['Heart Rate Variability'] = HeartRateVariability(vf).values  
            #     elif algorithm =='Blood Pressure Variability':
            #         results['Blood Pressure Variability'] = BloodPressureVariability(vf).values
            #     elif algorithm =='BRS':
            #         results['BRS'] = BRS(vf).values
            #     elif algorithm =='RSA:
            #         results['RSA'] = RSA(vf).values
                
                 # Pendiente añadir Variables autonomicas
                 elif algorithm == 'ICP Model':
                    results['ICP Model'] = icp_model() #Pendiente ver como añadir el modelo de ICP
                 elif algorithm == 'ABP Model':
                    results['ABP Model'] = abp_model() #Pendiente ver como añadir el modelo de ABP

            #Pendiente exportar los resultados a csv o a un vitalfile o hacer algo con ellos
        
        elif modo == "online":
            try: 
                print("Modo online activado. Iniciando monitoreo y ejecución de algoritmos...")
                import threading

                stop_event = threading.Event()
                streaming_thread = threading.Thread(target = main_loop, args=(stop_event,), name = "StreamingThread")

                print("-- Iniciando Streaming (Streaming_to_zarr.py) en thread separado --")
                streaming_thread.start()

                print("-- Iniciando bucle (zarr_to_algorithms.py) en Hilo Principal --")

                while True:
                    results = iniciar_streaming_en_thread() # Results es la variable que contiene un dataframe con todos los calculos hechos por algoritmos en el burst de datos (20-30 secs)
                    # Aqui hacemos la actualizacion al front

            except KeyboardInterrupt:
                print('\nInterrupción de usuario recibida. Deteniendo Streaming...')
                stop_event.set()
                streaming_thread.join()
                print('Streaming detenido. Programa finalizado.')
        else:
            print("Modo no reconocido. Saliendo.")
    except KeyboardInterrupt:
        print('\nInterrupción de usuario recibida. Saliendo...')


if __name__ == '__main__':
    main()
