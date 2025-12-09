import time
import threading

# --- IMPORTS ONLINE ---
#from Algorithms import ejecutar_algoritmos  # si lo necesitas en algún punto
from Streaming.Streaming_to_zarr import main_loop
from Streaming.zarr_to_algorithms import main_to_loop
from Zarr.utils_zarr_corrected import (
    ALGORITMOS_VISIBLES,
    STORE_PATH,
    escribir_prediccion,
)
from Front.Interface_online import RealTimeApp

# --- IMPORTS OFFLINE (ajusta a tus módulos reales si cambian nombres) ---
# from Algorithms.check_availability import check_availability
# from SomeModule import VitalFile, find_latest_vital
# from Algorithms.shock_index import ShockIndex
# ...


def seleccionar_modo_gui():
    """Muestra una pequeña ventana para elegir modo (online/offline).
    Si no hay Tk o hay error, cae a input() por consola."""
    try:
        import tkinter as tk
    except Exception:
        try:
            return input("Selecciona modo (online/offline): ")
        except Exception:
            return None

    selection = {'mode': None}
    root = tk.Tk()
    root.title('Seleccionar modo')
    root.resizable(False, False)

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

    btn_off = tk.Button(frm, text='Offline', width=10,
                        command=lambda: set_mode('offline'))
    btn_off.pack(side='left', padx=10)

    btn_on = tk.Button(frm, text='Online', width=10,
                       command=lambda: set_mode('online'))
    btn_on.pack(side='left', padx=10)

    root.protocol('WM_DELETE_WINDOW', lambda: set_mode(None))

    try:
        root.mainloop()
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    return selection['mode']


def iniciar_streaming_en_thread(algoritmes_escollits):
    """Llama a main_to_loop(algoritmes_escollits) para obtener resultados de algoritmos
    y los escribe en Zarr mediante escribir_prediccion(). Devuelve results."""
    results = main_to_loop(algoritmes_escollits)

    for NombreAlgoritmo, df_result in results.items():
        value_columns = [
            col for col in df_result.columns
            if col not in ('Timestamp', 'Time_ini_ms', 'Time_fin_ms')
        ]

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
                escribir_prediccion(
                    STORE_PATH,
                    track_name,
                    time_ms_array,
                    value_array,
                    model_info={"model": NombreAlgoritmo,
                                "visibilidad": visible},
                )
            else:
                escribir_prediccion(
                    STORE_PATH,
                    track_name,
                    time_ini_ms_array,
                    value_array,
                    model_info={"model": NombreAlgoritmo,
                                "visibilidad": visible},
                    timestamps_fin_ms=time_fin_ms_array,
                )

    return results


def modo_offline():
    """Aquí va tu lógica offline (VitalFile, check_availability, etc.)."""
    print("Modo offline activado. Esperando acciones... (Ctrl+C para salir de offline)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Saliendo de modo offline.")


def modo_online():
    """Arranca streaming + algoritmos + interfaz gráfica RealTimeApp."""
    print("Modo online activado. Iniciando monitoreo y ejecución de algoritmos...")

    stop_event = threading.Event()
    streaming_thread = threading.Thread(
        target=main_loop,
        args=(stop_event,),
        name="StreamingThread",
        daemon=True,
    )

    print("-- Iniciando Streaming (Streaming_to_zarr.py) en thread separado --")
    streaming_thread.start()

    # Algoritmos escogidos (por ahora puedes poner una lista fija o integrarlo con la GUI)
    algoritmes_escollits = []  # TODO: enlazar con lo que elija el usuario en la interfaz

    # Lanza la interfaz en el hilo principal
    app = RealTimeApp()

    # Ejemplo: actualizar resultados de algoritmos periódicamente
    def actualizar_resultados():
        # results = iniciar_streaming_en_thread(algoritmes_escollits)
        # TODO: pasar 'results' a app para que actualice sus gráficas con datos reales
        app.after(20000, actualizar_resultados)  # cada 20 s (ajusta)

    app.after(20000, actualizar_resultados)

    try:
        app.mainloop()
    finally:
        print("Cerrando interfaz. Deteniendo streaming...")
        stop_event.set()
        streaming_thread.join()
        print("Streaming detenido.")


def main():
    """Bucle principal: muestra el selector, entra en offline/online,
    y al terminar vuelve al selector salvo que el usuario cierre."""
    try:
        while True:
            modo = seleccionar_modo_gui()

            if modo is None:
                print("No se seleccionó modo. Saliendo.")
                break

            if modo == "offline":
                modo_offline()

            elif modo == "online":
                modo_online()
                print("Interfaz online cerrada. Volviendo al selector.")

            else:
                print("Modo no reconocido. Saliendo.")
                break

    except KeyboardInterrupt:
        print("\nInterrupción de usuario recibida. Saliendo...")


if __name__ == '__main__':
    main()
