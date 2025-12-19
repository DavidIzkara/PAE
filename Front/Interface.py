import matplotlib
matplotlib.use('TkAgg')

import tkinter as tk
from tkinter import ttk
from collections import deque
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from Zarr.utils_zarr_corrected import epoch1700_to_datetime


class RealTimeApp(tk.Tk):
    def __init__(self, available_algorithms_list, session_info="", config_queue = None):
        # Configuracion de Ventana
        super().__init__()              
        self.config_queue = config_queue
        self.title("Visualización en Tiempo Real")  
        self.geometry("1650x850")

        self.data_buffers = {}
        self.MAX_PUSHES = 10

        # Configuracion de Algoritmos (en el combobox)
        self.DEFAULT_OPTION = "--Seleccione algoritmo--"
        if not available_algorithms_list:   
            self.available_algorithms = [self.DEFAULT_OPTION]       
        else:
            self.available_algorithms = [self.DEFAULT_OPTION] + available_algorithms_list

        self._stop_flag = False
        
        # Configuracion Header
        self.info_header = tk.Label(self, text = session_info, font = ("Arial", 12, "bold")) 
        self.info_header.pack(pady=5)   

        selector_frame = tk.Frame(self)
        selector_frame.pack(pady=1)

        self.current_selections = [] # Llista que guarda els algoritmes selsecionats
        self.combos = [] # Llista que guarda els combos
        
        for i in range(4):
            col_frame = tk.Frame(selector_frame)
            col_frame.grid(row=0, column=i, padx=2)

            ttk.Label(col_frame, text=f"Grafica {i+1}:", font=("Arial", 10, "bold")).pack()

            combo = ttk.Combobox(col_frame, values=self.available_algorithms, state="readonly", width=30)
            combo.pack(pady=2)

            combo.set(self.DEFAULT_OPTION)

            combo.bind("<<ComboboxSelected>>", self.on_combo_changed)
            
            self.combos.append(combo)
            self.current_selections.append(self.DEFAULT_OPTION)

        self.fig, self.axes = plt.subplots(4, 1, figsize=(16, 8), sharex=True)
        self.fig.subplots_adjust(hspace=0.4, left=0.05, right=0.95, top=0.95, bottom=0.1)

        for i, ax in enumerate(self.axes):
            ax.set_title(f"Grafica {i+1}: {self.DEFAULT_OPTION}")
            ax.set_facecolor('#f9f9f9')
            ax.grid(True, linestyle='--', alpha=0.6)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_combo_changed(self, event):
        new_selections = [c.get() for c in self.combos]
        self.current_selections = new_selections

        print(f"- Cambio en selectores: {self.current_selections}")

        if self.config_queue:
            self.config_queue.put(self.current_selections)

    def update_data_and_plots(self, prediction_dict):
        
        if self._stop_flag or not prediction_dict:
            return
        
        for algo_name, df_nuevo in prediction_dict.items():
            if algo_name not in self.data_buffers:
                self.data_buffers[algo_name] = deque(maxlen=self.MAX_PUSHES)

            self.data_buffers[algo_name].append(df_nuevo)

        global_x_min = None
        global_x_max = None

        for i, (algo_name, ax) in enumerate(zip(self.current_selections, self.axes)):

            ax.clear()
            ax.grid(True, linestyle=':', alpha=0.9)

            ax.margins(x=0)

            if algo_name == self.DEFAULT_OPTION:
                ax.set_title(f"Grafica {i+1}: Sin selección")
                ax.set_facecolor('#f0f0f0') 
                continue

            if algo_name in prediction_dict:
                lista_bloques = list(self.data_buffers[algo_name])
                if not lista_bloques: continue
                
                df_ventana = pd.concat(lista_bloques, ignore_index=True)
                if df_ventana.empty: continue

                value_cols = [c for c in df_ventana.columns if c not in ['Timestamp', 'Time_fin_ms']]
                if not value_cols: continue
                
                target_col = value_cols[0]
                ax.set_facecolor('white')
                
                x_values = [epoch1700_to_datetime(t/1000) for t in df_ventana['Timestamp']]
                y_values = df_ventana[target_col].values

                if global_x_min is None or x_values[0] < global_x_min: global_x_min = x_values[0]
                if global_x_max is None or x_values[-1] > global_x_min: global_x_min = x_values[-1]

                ax.plot(x_values, y_values, label=f"{target_col}", color='tab:blue', linewidth=1.5)

                last_x, last_y = x_values[-1], y_values[-1]
                ax.plot(last_x, last_y, marker='o', color='red', markersize=6)
                ax.annotate(f'{last_y:.2f}', xy=(last_x, last_y), xytext=(8, 0), textcoords='offset points', color='red', weight='bold')

                ax.set_title(f"{algo_name} - {target_col}", fontsize=11, fontweight='bold')
                ax.legend(loc='upper left')
        
        if global_x_max and global_x_min:
            self.axes[3].set_xlim(global_x_min, global_x_max)
        
        self.axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
        self.axes[3].xaxis.set_major_locator(mdates.AutoDateLocator())

        self.axes[3].tick_params(axis='x', rotation=0, labelsize=10, labelbottom=True)

        self.canvas.draw()

    def on_close(self):
        print("- Cerrando interfaz grafica... ")
        self._stop_flag = True
        try:
            plt.close('all')
            self.quit()
            self.destroy()
        except Exception as e:
            print(f"- Error al cerrar: {e}")

if __name__ == "__main__":
    app = RealTimeApp(["Shock Index", "Heart Rate Variability", "RSA"], "SESION_TEST_01")
    app.mainloop()
