import matplotlib
matplotlib.use('TkAgg')

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from Zarr.utils_zarr_corrected import epoch1700_to_datetime


class RealTimeApp(tk.Tk):
    def __init__(self, available_algorithms_list, session_info="", config_queue = None):
        super().__init__()
        self.config_queue = config_queue
        self.title("Visualización en Tiempo Real")
        self.geometry("1650x850")

        self.available_algorithms = available_algorithms_list
        
        if not self.available_algorithms:
            self.available_algorithms = ["Sin algoritmos disponibles"]

        self.info_header = tk.Label(self, text = session_info, font = ("Arial", 12, "bold"))
        self.info_header.pack(pady=5)

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

        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 6.5), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.canvas.mpl_connect("button_press_event", self.on_graph_click)

        self.text_annotations = [None, None, None, None]

        self.algorithms_results_data = {}
        # id del callback `after` para poder cancelarlo al cerrar
        self._after_id = None
        self.draw_plots()

        self._stop_flag = False
        self.refresh_loop()
        # Captura cierre de la ventana principal para parar el loop
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_data_and_plots(self, algorithms_results: dict):
        self.algorithms_results_data = algorithms_results
        self.draw_plots(use_real_data=True)

    def on_combobox_change(self, event, index):
        current_combo = self.selected_vars[index]
        new_value = current_combo.get()

        previous_value = self.current_selections[index]

        if new_value != previous_value:
            self.current_selections[index] = new_value
            
            if self.config_queue:
                self.config_queue.put(self.current_selections.copy())
            print(f"- Combobox {index+1} cambió de '{previous_value}' a '{new_value}'.")            
            print(f"- self.current_selections ahora es: {self.current_selections}")

            self.draw_plots()
        else:
            print(f"- Combobox {index+1} seleccionado, pero el valor sigue siendo '{new_value}'. No se hace nada.")

    def draw_plots(self, use_real_data = False):
        for i in range(2):
            for j in range(2):
                
                idx = i * 2 + j
                ax = self.axs[i, j]
                ax.clear()
                algo_name = self.selected_vars[idx].get()
                ax.set_title(algo_name)
                
                if use_real_data and algo_name in self.algorithms_results_data:
                    df_result = self.algorithms_results_data[algo_name]
                    if not df_result.empty:
                        # Asumimos que el DataFrame tiene 'Timestamp' (o 'Time_ini/fin_ms') y una columna de valor
                        if 'Timestamp' in df_result.columns and len(df_result.columns) > 1:
                            raw_x = df_result['Timestamp'].values

                            x = [epoch1700_to_datetime(ts/1000) for ts in raw_x] # el calculo que hace es para pasar de milisegundos a segundos (como espera la funcion)                      
                        else: # 'Time_ini/fin_ms'
                            raw_x_ini = df_result['Time_ini_ms'].values
                            raw_x_fin = df_result['Time_fin_ms'].values
                            
                            x = [epoch1700_to_datetime(ts/1000) for ts in raw_x_ini]
                            x_fin = [epoch1700_to_datetime(ts/1000) for ts in raw_x_fin] # Nose que valor de timestamp interesa mas mostrar si inicio o final.

                        value_col = [col for col in df_result.columns if col not in ['Timestamp', 'Time_ini_ms', 'Time_fin_ms']][0]
                        y = df_result[value_col].values

                        ax.plot(x, y, marker = '.', linestyle = '-')
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))

                        last_x = x[-1]
                        last_y = y[-1]

                        ax.text(last_x, last_y, f" {last_y:.2f}", color='red', weight = 'bold', verticalalignment = 'center', bbox = dict(facecolor = 'white', alpha = 0.6, edgecolor = 'none'))
                    else:
                        self._plot_mockup(ax, algo_name, i, j)
                else:
                    self._plot_mockup(ax, algo_name, i, j)

        self.canvas.draw()

    def _plot_mockup(self, ax, algo_name, i, j):
        """
        Dibuja datos simulados (mock-up) para mantener el gráfico visible y responsivo.
        La forma del mock-up depende del nombre del algoritmo.
        """
        import numpy as np # Necesitarás este import si no lo tienes globalmente
        
        # Datos base de X
        x = np.linspace(0, 5, 50)
        base_factor = i * 2 + j + 1 # Factor único basado en la posición del subplot (1 a 4)

        if "Shock" in algo_name:
            # Simula un crecimiento lineal
            factor = base_factor * 1.0
            y = factor * x + np.random.normal(0, 0.1, size=x.size)
            ax.plot(x, y, color='blue', linestyle='--')
            
        elif "Driving" in algo_name:
            # Simula un crecimiento exponencial o curvo
            factor = base_factor * 1.5
            y = factor * (x ** 1.2) + np.random.normal(0, 0.5, size=x.size)
            ax.plot(x, y, color='red', linestyle='-')
            
        elif "Hypoperfusion" in algo_name:
            # Simula una caída y luego un rebote (forma de U invertida)
            factor = base_factor * 2.0
            y = factor * np.sin(x / 2) + factor + np.random.normal(0, 0.2, size=x.size)
            ax.plot(x, y, color='green', marker='v', markevery=10)
            
        elif "Sepsis" in algo_name:
            # Simula una serie de pasos o valores constantes (alarma)
            factor = base_factor * 0.5
            y = np.repeat([factor, factor * 1.2, factor, factor * 1.4], x.size // 4)
            # Asegurar que el tamaño coincida
            if y.size < x.size:
                 y = np.concatenate([y, [y[-1]] * (x.size - y.size)]) 
            elif y.size > x.size:
                 y = y[:x.size]

            ax.step(x, y, where='post', color='purple')

        else:
            # Caso por defecto (crecimiento lineal simple)
            factor = base_factor
            y = factor * x + np.random.normal(0, 0.05, size=x.size)
            ax.plot(x, y, marker='o', markersize=3, linestyle='-', alpha=0.6)
            
        ax.set_ylim(0, max(y) * 1.5 if y.size > 0 else 10)

    def refresh_loop(self):
        # Si se ha marcado para parar, no hacer nada ni re-programar
        if getattr(self, "_stop_flag", True):
            return
        try:
            pass
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
