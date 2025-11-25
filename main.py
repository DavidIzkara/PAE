import time
from Streaming import monitorizar_actualizacion_recurso
from Algorithms import ejecutar_algoritmos
from Interface import actualizar_interfaz


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


def main():
    modo = seleccionar_modo_gui()

    if modo is None:
        print("No se seleccionó modo. Saliendo.")
        return

    try:
        if modo == "offline":
            print("Modo offline activado. Esperando acciones...")
            while True:
                time.sleep(1)
        elif modo == "online":
            print("Modo online activado. Iniciando monitoreo y ejecución de algoritmos...")
            while True:
                monitorizar_actualizacion_recurso()
                ejecutar_algoritmos()
                actualizar_interfaz()
                time.sleep(1)  # Para evitar que el bucle consuma mucho CPU
        else:
            print("Modo no reconocido. Saliendo.")
    except KeyboardInterrupt:
        print('\nInterrupción de usuario recibida. Saliendo...')


if __name__ == '__main__':
    main()
