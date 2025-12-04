import matplotlib
matplotlib.use('TkAgg')

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Importa función de Algorithms
from Algorithms.check_availability import check_availability


class RealTimeApp(tk.Tk):
    def __init__(self, available_algorithms_list):
        super().__init__()
        self.title("Visualización en Tiempo Real")
        self.geometry("1650x850")

        # ------------------------------------------------------------------
        # 1) Obtener las tracks disponibles (AQUÍ LUEGO PONDRÁS TUS TRACKS REALES)
        #   Por ahora, ejemplo mínimo con algunas keys para que veas que funciona.
        #   Sustituye este dict por el que te llegue de Streaming/Zarr/etc.
        # ------------------------------------------------------------------
        # Ejemplo de tracks SIMPLIFICADOS (sin prefijo vendor) —
        # coinciden con las keys que espera `check_availability`.
        example_tracks = [
            "ECG_HR",
            "ABP_SYS",
            "PPLAT_CMH2O",
            "PEEP_CMH2O",
            "TV_EXP",
            "PIP_CMH2O",
            "PLETH_SAT_O2",
            "FiO2",
        ]

        # 2) Llamar a check_availability para saber qué algoritmos se pueden calcular
        self.available_algorithms = available_algorithms_list##check_availability(example_tracks)
        # Si por lo que sea no hay ninguno, evita que los combo queden vacíos
        if not self.available_algorithms:
            self.available_algorithms = ["Sin algoritmos disponibles"]

        # ------------------------------------------------------------------
        # 3) Construir interfaz de selección usando available_algorithms
        # ------------------------------------------------------------------
        selector_frame = tk.Frame(self)
        selector_frame.pack(pady=1)

        self.current_selections = []
        self.selected_vars = []
        for i in range(4):
            label = ttk.Label(selector_frame, text=f"Algoritmo gráfica {i+1}:")
            label.grid(row=0, column=i, padx=2)

            combo = ttk.Combobox(
                selector_frame,
                values=self.available_algorithms,
                state="readonly",
                width=22,
            )
            combo.grid(row=1, column=i, padx=2)

            # Valor por defecto: primer algoritmo disponible
            combo.current(0)
            self.selected_vars.append(combo)
            self.current_selections.append(combo.get())

            combo.bind("<<ComboboxSelected>>", lambda e, index=i: self.on_combobox_change(event = e, index = index))
            

        # Botón visible para volver al selector de modo sin cerrar la ventana
        btn_back = tk.Button(selector_frame, text='Volver', width=12, command=self.on_close)
        # Colocamos el botón a la derecha de los combos
        btn_back.grid(row=0, column=4, rowspan=2, padx=8, pady=2, sticky='ns')

        plot_frame = tk.Frame(self)
        plot_frame.pack(fill="both", expand=True)

        self.fig, self.axs = plt.subplots(
            2, 2, figsize=(12, 6.5), constrained_layout=True
        )
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.canvas.mpl_connect("button_press_event", self.on_graph_click)

        self._stop_flag = False
        # id del callback `after` para poder cancelarlo al cerrar
        self._after_id = None
        self.draw_plots()
        self.refresh_loop()
        # Captura cierre de la ventana principal para parar el loop
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.quit()

    def on_combobox_change(self, event, index):

        current_combo = self.selected_vars[index]
        new_value = current_combo.get()

        previous_value = self.current_selections[index]

        if new_value != previous_value:
            self.current_selections[index] = new_value

            print(f"✅ Combobox {index+1} cambió de '{previous_value}' a '{new_value}'.")            
            print(f"   self.current_selections ahora es: {self.current_selections}")

            self.draw_plots()
            self.quit()
        else:
            print(f"⚠️ Combobox {index+1} seleccionado, pero el valor sigue siendo '{new_value}'. No se hace nada.")

    def draw_plots(self):
        for i in range(2):
            for j in range(2):
                ax = self.axs[i, j]
                ax.clear()

                algo_name = self.selected_vars[i * 2 + j].get()

                # ----------------------------------------------------------
                # AQUÍ es donde enlazarás con cada algoritmo real.
                # De momento, simplemente cambiamos la pendiente/forma para
                # visualizar que cada curva está asociada a un algoritmo.
                # ----------------------------------------------------------
                x = [1, 2, 3, 4, 5]

                # Ejemplo muy simple: factor según el índice + algo por nombre
                base_factor = i * 2 + j + 1
                # Diferenciamos visualmente por el nombre del algoritmo
                if "Shock" in algo_name:
                    # Monótono linear
                    factor = base_factor * 1.0
                    y = [factor * v for v in x]
                elif "Driving" in algo_name:
                    # Más pronunciado
                    factor = base_factor * 1.5
                    y = [factor * v ** 1.2 for v in x]
                elif "Dynamic" in algo_name:
                    # Curva menos pronunciada
                    factor = base_factor * 0.8
                    y = [factor * (v ** 0.9) for v in x]
                elif "ROX" in algo_name or "ROX Index" in algo_name:
                    # Curva con pequeño decrecimiento
                    factor = base_factor * 1.1
                    y = [factor * (6 - v) * 0.5 for v in x]
                elif "Temp" in algo_name:
                    # Oscilación pequeña (ejemplo de comparación térmica)
                    factor = base_factor * 0.6
                    y = [factor * (1 + 0.2 * ((-1) ** v)) for v in x]
                elif "Cardiac" in algo_name:
                    # Otra forma para diferenciar
                    factor = base_factor * 1.3
                    y = [factor * (v + 0.5 * (i + j)) for v in x]
                else:
                    # Default: lineal simple
                    factor = base_factor
                    y = [factor * v for v in x]

                ax.plot(x, y, marker='o')
                ax.set_title(algo_name)

        self.canvas.draw()

    def refresh_loop(self):
        # Si se ha marcado para parar, no hacer nada ni re-programar
        if getattr(self, "_stop_flag", True):
            return
        try:
            self.draw_plots()
        except Exception:
            pass
        try:
            # Guardamos el id para poder cancelarlo en on_close
            self._after_id = self.after(1000, self.refresh_loop)  # cada segundo
        except Exception:
            self._after_id = None

    def on_graph_click(self, event):
        if event.inaxes:
            for i in range(2):
                for j in range(2):
                    if event.inaxes == self.axs[i, j]:
                        self.maximize_plot((i, j))

    def maximize_plot(self, idx):
        new_win = tk.Toplevel(self)
        new_win.title(f"Gráfica {idx[0] * 2 + idx[1] + 1} - Ampliada")

        fig, ax = plt.subplots(figsize=(10, 8))
        algo_name = self.selected_vars[idx[0] * 2 + idx[1]].get()

        x = [1, 2, 3, 4, 5]
        base_factor = idx[0] * 2 + idx[1] + 1
        if "Shock" in algo_name:
            factor = base_factor * 1.0
        elif "Driving" in algo_name:
            factor = base_factor * 1.5
        elif "Dynamic" in algo_name:
            factor = base_factor * 0.8
        else:
            factor = base_factor
        y = [factor * v for v in x]

        ax.plot(x, y, marker='o')
        ax.set_title(algo_name)

        canvas = FigureCanvasTkAgg(fig, master=new_win)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def on_close(self):
        # Señala para parar el loop de refresco y destruye la ventana.
        # Cuando se destruye la instancia, `mainloop()` que estaba en
        # `main_visual.py` regresará y el programa podrá volver a mostrar
        # el selector de modo.
        try:
            # Marcar para detener refrescos
            self._stop_flag = True
            # Cancelar callback pendiente si existe
            if getattr(self, "_after_id", None):
                try:
                    self.after_cancel(self._after_id)
                except Exception:
                    pass
            # Intentar terminar la mainloop si está corriendo
            try:
                self.quit()
            except Exception:
                pass
            # Cerrar todas las figuras de matplotlib para evitar callbacks
            try:
                plt.close('all')
            except Exception:
                pass
            # Procesar tareas pendientes del intérprete Tk
            try:
                self.update_idletasks()
            except Exception:
                pass
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass
