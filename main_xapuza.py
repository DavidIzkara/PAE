import time
import threading 
import os
import queue
import tkinter as tk
from Front.Interface import RealTimeApp
from Streaming.Streaming_to_zarr import main_loop, PRUEVAS, DIRECTORIO_PRUEVA, ARCHIVO_VITAL, BASE_DIR, seleccionar_base_dir
from Streaming.zarr_to_algorithms import main_to_loop, reset_last_processed_timestamp, inicialitzar_algoritmes_amb_buffer
from Streaming.utils_Streaming import obtener_directorio_del_dia, obtener_vital_mas_reciente
from Zarr.utils_zarr_corrected import VISIBLE_ALGORITHMS, STORE_PATH, write_prediction


def seleccionar_modo_gui():

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

def seleccionar_box():
    selection = {'box': None}
    root = tk.Tk()
    root.title('Seleccionar BOX')
    root.resizable(False, False)

    width, height = 280, 140
    
    try:
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        x = int((sw - width) / 2)
        y = int((sh - height) / 2)
        root.geometry(f"{width}x{height}+{x}+{y}")
    except Exception:
        root.geometry(f"{width}x{height}")

    tk.Label(root, text='Introduce el numero del BOX (1-13):').pack(pady=(12, 4))
    entry = tk.Entry(root, justify='center')
    entry.insert(0, "1")
    entry.pack(pady=(0, 8))

    def acceptar():
        val = entry.get().strip()
        if not val.isdigit():
            entry.delete(0, tk.END); entry.insert(0, "1"); return
        n = int(val)
        if n < 1 or n > 13:
            entry.delete(0, tk.END); entry.insert(0, "1"); return
        selection['box'] = n
        root.quit()

    def cancelar():
        selection['box'] = None
        root.quit()
    
    frm = tk.Frame(root); frm.pack(pady=(0, 8))
    tk.Button(frm, text='OK', width=8, command=acceptar).pack(side='left', pady=6)
    tk.Button(frm, text='Cancelar', width=8, command=cancelar).pack(side='left', pady=6)

    root.protocol('WM_DELETE_WINDOW', cancelar)
    try:
        root.mainloop()
    finally:
        try: root.destroy()
        except Exception: pass

    return selection['box']

def construir_zarr_con_box_y_uid(box: int, uid: str, store_path_actual:str) -> str:

    base_dir, basename = os.path.split(store_path_actual)
    
    zarr_dir = base_dir if basename.endswith('.zarr') else store_path_actual
    
    nombre = f"BOX{box:02d}_{uid}.zarr"
    
    return os.path.join(zarr_dir, nombre)


def extraer_uid_de_vital_path(vital_path: str) -> str:

    base = os.path.basename(vital_path)
    base = base.replace('.vital', '')

    return base.split('_')[0] if '_' in base else base


def actualizar_store_path_global(nuevo_store_path: str):

    global STORE_PATH
    STORE_PATH = nuevo_store_path

    import Zarr.utils_zarr_corrected as zarr_utils
    zarr_utils.STORE_PATH = nuevo_store_path

    import Streaming.Streaming_to_zarr as stz_module
    stz_module.STORE_PATH = nuevo_store_path

    import Streaming.zarr_to_algorithms as z2a_module
    z2a_module.STORE_PATH = nuevo_store_path

    import Front.Interface as front_interface
    front_interface.STORE_PATH = nuevo_store_path

    print(f"- Nuevo STORE_PATH aplicado: {nuevo_store_path}")

def data_processing_loop(app, config_queue, streaming_control):
   
    print("- Iniciando Bucle de Procesamiento de Datos (zarr_to_algorithms) en thread separado --")
    
    ultima_seleccion = [alg.get() if hasattr(alg, 'get') else alg for alg in app.current_selections]
    
    box_num = streaming_control.get('box_num')

    while not streaming_control['stop_signal'].is_set():
        try:
            
            if not PRUEVAS:
                import Streaming.Streaming_to_zarr as stz_module
                nueva_carpeta = obtener_directorio_del_dia(stz_module.BASE_DIR)
                nuevo_path = obtener_vital_mas_reciente(nueva_carpeta)

                if nuevo_path and nuevo_path != streaming_control['current_path']:
                    print(f"\n--- DETECTADO CAMBIO DE ARCHIVO .vital: {os.path.basename(nuevo_path)} ---")

                    
                    if box_num is not None:
                        try:
                            import Zarr.utils_zarr_corrected as zarr_utils
                            uid_nuevo = extraer_uid_de_vital_path(nuevo_path)
                            nuevo_store = construir_zarr_con_box_y_uid(box_num, uid_nuevo, zarr_utils.STORE_PATH)
                            actualizar_store_path_global(nuevo_store)
                            print(f"--- 🆕 ZARR seleccionado por cambio de .vital: {nuevo_store}")
                        except Exception as e:
                            print(f"--- [WARN] No se pudo actualizar STORE_PATH con nuevo uid: {e}")


                    # Paramos el hilo de streaming actual
                    streaming_control['stop_signal'].set()
                    streaming_control['thread'].join(timeout=2)

                    reset_last_processed_timestamp()

                    # Crea el nuevo hilo
                    new_stop_signal = threading.Event()
                    new_algoritmos_disponibles = []
                    new_algoritmos_cargados_event = threading.Event()
                    new_thread = threading.Thread(target=main_loop, args=(new_stop_signal, new_algoritmos_cargados_event, new_algoritmos_disponibles, nuevo_path, nueva_carpeta), name="StreamingThread")
                    
                    # Guarda los parametros en el streaming_control y lanzalo
                    streaming_control['thread'] = new_thread
                    streaming_control['current_path'] = nuevo_path
                    streaming_control['stop_signal'] = new_stop_signal

                    new_thread.start()

                    print(f"--- ✅ Streaming reiniciado con el nuevo archivo ---\n")

            canvio_detectado = False
            while not config_queue.empty():
                try:
                    ultima_seleccion = config_queue.get_nowait()
                    canvio_detectado = True
                    reset_last_processed_timestamp()
                except queue.Empty:
                    break
            
            if canvio_detectado:
                print(f"- Saltando a la configuración más reciente: {ultima_seleccion}")
        
            results = main_to_loop(ultima_seleccion, stop_event=streaming_control['stop_signal'])
            
            if results:
                print(f"[- {threading.current_thread().name}] Resultados obtenidos. Escribiendo predicciones...")

                for NombreAlgoritmo, df_result in results.items():
                    value_columns = [col for col in df_result.columns if col != 'Timestamp' and col != 'Time_ini_ms' and col != 'Time_fin_ms']
                    
                    if 'Timestamp' in df_result.columns:
                        time_ini_array = df_result['Timestamp'].values
                        time_fin_array = None
                    elif 'Time_ini_ms' in df_result.columns and 'Time_fin_ms' in df_result.columns:
                        time_ini_array = df_result['Time_ini_ms'].values
                        time_fin_array = df_result['Time_fin_ms'].values
                    else:
                        continue

                    visible = NombreAlgoritmo in VISIBLE_ALGORITHMS

                    for track_name in value_columns:
                        value_array = df_result[track_name].values
                        write_prediction(zarr_path=STORE_PATH, pred_name=track_name, timestamps_ms=time_ini_array, values=value_array, modelo_info = {"model": NombreAlgoritmo, "visibilidad": visible}, timestamps_fin_ms=time_fin_array)
                        
                app.after(0, app.update_data_and_plots, results)

                print(f"- [{threading.current_thread().name}] Solicitada actualización de GUI.")
            
            time.sleep(0.01) # Esto permite a el front reaccionar cada 0.5 s i no cada 30 s

        except Exception as e:
            print(f"- [ERROR] {e}")
            time.sleep(1)

def main():
    modo = seleccionar_modo_gui()

    if modo is None:
        print("- No se seleccionó modo. Saliendo.")
        return

    try:
        if modo == "online":
            try:
                print("- Modo online activado. Iniciando monitoreo y ejecución de algoritmos...")

                box_num = seleccionar_box()
                if box_num is None:
                    print("- Selección de box cancelada. Saliendo. -")
                    return

                if not PRUEVAS:
                    base_dir_seleccionado = seleccionar_base_dir()
                    if not base_dir_seleccionado:
                        print("- No se seleccionó carpeta base. Saliendo. -")
                        return
                    
                    import Streaming.Streaming_to_zarr as stz_module
                    stz_module.BASE_DIR = base_dir_seleccionado
                    print(f"- BASE_DIR seleccionado: {base_dir_seleccionado}")

                if PRUEVAS:
                    directorio_dia = DIRECTORIO_PRUEVA
                    vital_path = os.path.join(directorio_dia, ARCHIVO_VITAL)
                    vital_path_con_ext = vital_path + ".vital"
                else: # En caso real
                    try: 
                        import Streaming.Streaming_to_zarr as stz_module
                        directorio_dia = obtener_directorio_del_dia(stz_module.BASE_DIR)
                        vital_path_con_ext = obtener_vital_mas_reciente(directorio_dia)
                        vital_path = vital_path_con_ext.replace(".vital", "")
                    except FileNotFoundError as e:
                        print(f"- Error: {e}")
                        return
                
                # vital_path = bd9cftsa6_251205_140000
                partes = vital_path.replace('.vital', '').split('_')
                #uid = partes[0] # bd9cftsa6 (uid del paciente, en este caso the boss xD)
                #fecha_str = partes[1] # 251205
                #hora_str = partes[2] # 140000

                #Formateamos para que sea comprensible para el humano
                fecha_txt = f"Inicio: {partes[1][4:6]}/{partes[1][2:4]}/20{partes[1][0:2]} - {partes[2][0:2]}"

                uid = extraer_uid_de_vital_path(vital_path_con_ext if not PRUEVAS else (vital_path + ".vital"))
                import Zarr.utils_zarr_corrected as zarr_utils
                nuevo_store_path = construir_zarr_con_box_y_uid(box_num, uid, zarr_utils.STORE_PATH)

                print(f"- BOX seleccionado: {box_num:02d}")
                print(f"- UID detectado: {uid}")
                print(f"- Nuevo ZARR: {nuevo_store_path}")

                actualizar_store_path_global(nuevo_store_path)                

                inicialitzar_algoritmes_amb_buffer()
                
                stop_event = threading.Event()
                algoritmos_disponibles = []
                algoritmos_cargados_event = threading.Event()

                streaming_thread = threading.Thread(target=main_loop, args=(stop_event, algoritmos_cargados_event, algoritmos_disponibles, vital_path_con_ext, directorio_dia), name="StreamingThread", daemon=True)

                print("- Iniciando Streaming (Streaming_to_zarr.py) en thread separado --")
                streaming_thread.start()

                streaming_control = {'thread': streaming_thread, 'current_path': vital_path_con_ext, 'stop_signal': stop_event, 'box_num': box_num}

                print("- Esperando la lista de algoritmos disponibles del Streaming... --")
                algoritmos_cargados_event.wait(timeout=10)

                print("- Ya ha llegado la lista de algoritmos disponibles del Streaming --")
                if not algoritmos_disponibles:
                    # Lógica de manejo de error si no se cargan a tiempo
                    print("- Error: No se pudo obtener la lista de algoritmos. Saliendo.")
                    stop_event.set()
                    streaming_thread.join()
                    return

                paquete = algoritmos_disponibles[0] if isinstance(algoritmos_disponibles[0], dict) else {"all": algoritmos_disponibles, "visible": algoritmos_disponibles}
                lista_all = paquete.get("all", [])
                lista_visible = paquete.get("visible", [])

                config_queue = queue.Queue()

                app = RealTimeApp(available_algorithms_list_all=lista_all, available_algorithms_list_visible=lista_visible, session_info=fecha_txt, config_queue = config_queue)

                data_thread = threading.Thread(target = data_processing_loop, args = (app, config_queue, streaming_control), name = "DataProcessingThread", daemon=True)
                data_thread.start()
                
                app.mainloop()

                print('\n- Ventana de la GUI cerrada. Deteniendo threads...')
                
            except Exception as e:
                print(f"- Error crítico en modo online: {e}")
            
            finally:
                # Asegurar la detención de ambos threads
                stop_event.set()
                if 'streaming_thread' in locals() and streaming_thread.is_alive():
                    streaming_thread.join(timeout=2)
                if 'data_thread' in locals() and data_thread.is_alive():
                    data_thread.join(timeout=2)
                
                print('- Threads de procesamiento detenidos. Volviendo al selector de modo.')
        else:
            print("- Modo no reconocido o offline. Saliendo.")

    except KeyboardInterrupt:
        print('\n- Interrupción de usuario recibida. Saliendo...')

if __name__ == '__main__':
    main()
