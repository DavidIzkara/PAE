#!/usr/bin/env python3
import sys

def seleccionar_modo_gui():
    try:
        import tkinter as tk
    except Exception as e:
        print("No se pudo importar tkinter:", e)
        return None

    selection = {'mode': None}

    root = tk.Tk()
    root.title('Seleccionar modo')
    root.resizable(False, False)

    # Tamaño y centrado simple
    width = 320
    height = 140
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

    label = tk.Label(root, text='Selecciona modo:', font=("Segoe UI", 11))
    label.pack(pady=(14, 8))

    frm = tk.Frame(root)
    frm.pack()

    btn_off = tk.Button(frm, text='Offline', width=12, command=lambda: set_mode('offline'))
    btn_off.pack(side='left', padx=12)

    btn_on = tk.Button(frm, text='Online', width=12, command=lambda: set_mode('online'))
    btn_on.pack(side='left', padx=12)

    # Cerrar ventana = sin selección
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
        print("No se seleccionó modo (o tkinter no disponible).")
        sys.exit(0)
    print("Modo seleccionado:", modo)
    # Aquí solo mostramos la selección y salimos.
    # Si quieres mantener la app corriendo según el modo, reemplaza lo siguiente por la lógica que necesites.
    sys.exit(0)


if __name__ == '__main__':
    main()