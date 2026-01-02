import matplotlib
matplotlib.use('TkAgg')

import os
import tkinter as tk
from tkinter import ttk
from collections import deque
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import zarr
from Zarr.utils_zarr_corrected import epoch1700_to_datetime, STORE_PATH


class RealTimeApp(tk.Tk):
    def __init__(self, available_algorithms_list_all, available_algorithms_list_visible, session_info="", config_queue = None):
        # Configuracion de Ventana
        super().__init__()              
        self.config_queue = config_queue
        self.title("Visualización en Tiempo Real")  
        self.geometry("1650x850")

        self.data_buffers = {}
        self.MAX_PUSHES = 10

        # Configuracion de los algoritmos (en el combobox)
        self.DEFAULT_OPTION = "--Seleccione algoritmo--"
        self.available_algorithms_list_all = list(available_algorithms_list_all or [])
        self.available_algorithms_list_visible = list(available_algorithms_list_visible or [])

        self.available_algorithms = [self.DEFAULT_OPTION] + self.available_algorithms_list_visible

        self.is_researcher = tk.BooleanVar(value=False) # Checkbox de investigacion

        self.stop_flag = False

        # Configuracion Header
        self.info_header = tk.Label(self, text = session_info, font = ("Arial", 12, "bold")) 
        self.info_header.pack(pady=5)   

        selector_frame = tk.Frame(self)
        selector_frame.pack(pady=1)

        self.current_selections = [] # Llista que guarda els algoritmes selsecionats
        self.combos = [] # Llista que guarda els combos
        
        for i in range(4):
            var = tk.StringVar(value=self.DEFAULT_OPTION)
            self.current_selections.append(var)

        for i in range(4):
            col_frame = tk.Frame(selector_frame)
            col_frame.grid(row=0, column=i, padx=2)

            ttk.Label(col_frame, text=f"Grafica {i+1}:", font=("Arial", 10, "bold")).pack()

            combo = ttk.Combobox(col_frame, textvariable=self.current_selections[i], values=self.available_algorithms, state="readonly", width=30)
            combo.pack(pady=2)

            self.combos.append(combo)
        
        chk_frame = tk.Frame(selector_frame)
        chk_frame.grid(row=0, column=4, padx=(12, 0))
        chk = ttk.Checkbutton(chk_frame, text="Investigación", variable=self.is_researcher, command=self.on_researcher_toggeled)
        chk.pack(pady=(22, 0)) # Alineacion con los combobox

        for i in range(4):
            self.current_selections[i].trace_add("write", self.on_combo_changed)
            self.combos[i].bind("<<ComboboxSelected>>", self.update_all_combobox_lists)

        self.fig, self.axes = plt.subplots(4, 1, figsize=(16, 8), sharex=True)
        self.fig.subplots_adjust(hspace=0.4, left=0.05, right=0.95, top=0.95, bottom=0.1)

        for i, ax in enumerate(self.axes):
            ax.set_title(f"Grafica {i+1}: {self.DEFAULT_OPTION}")
            ax.set_facecolor('#f9f9f9')
            ax.grid(True, linestyle='--', alpha=0.6)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect('button_press_event', self.on_plot_click)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_all_combobox_lists(self, event=None):
        if len(self.combos) < 4: return
        
        # Elegir la fuente segun el checkbox
        if self.is_researcher.get():
            fuente = self.available_algorithms_list_all
        else: # checkbox = False
            fuente = self.available_algorithms_list_visible
        
        self.available_algorithms = [self.DEFAULT_OPTION] + fuente

        # Ajustar valores disponibles, evitando duplicados y preservando la selección actual si es válida
        seleccionados = [sv.get() for sv in self.current_selections if sv.get() != self.DEFAULT_OPTION]

        for i, combo in enumerate(self.combos):
            current_val = self.current_selections[i].get()

            opciones_libres = [self.DEFAULT_OPTION]
            for alg in self.available_algorithms:
                if alg == self.DEFAULT_OPTION: continue
                if alg not in seleccionados or alg == current_val:
                    opciones_libres.append(alg)
            
            combo['values'] = opciones_libres

            # Si al desmarcar "Investigación" el valor seleccionado ya no existe, devolver a DEFAULT
            if current_val != self.DEFAULT_OPTION and current_val not in opciones_libres:
                self.current_selections[i].set(self.DEFAULT_OPTION)

    def on_combo_changed(self, *args):
        current_config = [sv.get() for sv in self.current_selections]
        for i, sv in enumerate(self.current_selections):
            if sv.get() == self.DEFAULT_OPTION:
                self.clear_slot_buffer(i)
            else:
                self.clear_slot_buffer(i)

        if self.config_queue:
            self.config_queue.put(current_config)
        print(f"Nueva configuración enviada: {current_config}")

    def on_researcher_toggeled(self):
        # Recalcular listas y refrescar combobox
        self.update_all_combobox_lists()
        print(f"Modo Investigación: {self.is_researcher.get()}")

    def clear_slot_buffer(self, slot_index: int):
        try:
            algo_name = self.current_selections[slot_index].get()
            if algo_name and algo_name in self.data_buffers:
                self.data_buffers[algo_name].clear()
        
        except Exception:
            pass

    def update_data_and_plots(self, prediction_dict):
        
        if self.stop_flag:
            return
            
        temps_referencia_df = None
        if prediction_dict:
            for df in prediction_dict.values():
                if df is not None and not df.empty:
                    temps_referencia_df = df.copy()
                    break

        for i in range(4):
            algo_name = self.current_selections[i].get()
            if algo_name == self.DEFAULT_OPTION: continue

            if algo_name not in self.data_buffers:
                self.data_buffers[algo_name] = deque(maxlen=self.MAX_PUSHES)

            if prediction_dict and algo_name in prediction_dict and not prediction_dict[algo_name].empty:
                df_nuevo = prediction_dict[algo_name].copy()
                if 'Timestamp' not in df_nuevo.columns:
                    if 'Time_fin_ms' in df_nuevo.columns:
                        df_nuevo['Timestamp'] = df_nuevo['Time_fin_ms'].astype('int64')
                df_nuevo['real'] = True
                self.data_buffers[algo_name].append(df_nuevo)
            
            elif temps_referencia_df is not None and len(self.data_buffers[algo_name]) > 0:
                ultima_dada = self.data_buffers[algo_name][-1].copy()

                ultima_dada = ultima_dada.copy()

                ts_ref_col = None
                for c in ['Timestamp', 'time_ms', 'Time_fin_ms', 'Time_ini_ms']:
                    if c in temps_referencia_df.columns:
                        ts_ref_col = c
                        break
                
                n = min(len(ultima_dada), len(temps_referencia_df))
                if ts_ref_col is None or n == 0:
                    pass
                else:
                    ultima_dada = ultima_dada.iloc[:n].copy()
                    ts_value = temps_referencia_df[ts_ref_col].values[:n]

                    if 'Timestamp' in ultima_dada.columns:
                        ultima_dada.loc[:, 'Timestamp'] = ts_value
                    elif 'time_ms' in ultima_dada.columns:
                        ultima_dada.loc[:, 'time_ms'] = ts_value
                    elif 'Time_fin_ms' in ultima_dada.columns:
                        ultima_dada.loc[:, 'Time_fin_ms'] = ts_value
                    else:
                        ultima_dada.loc[:, 'Time_ini_ms'] = ts_value
                
                    if 'Time_ini_ms' in ultima_dada.columns and 'Time_ini_ms' in temps_referencia_df.columns:
                        ultima_dada.loc[:, 'Time_ini_ms'] = temps_referencia_df['Time_ini_ms'].values[:n]
                    if 'Time_fin_ms' in ultima_dada.columns and 'Time_fin_ms' in temps_referencia_df.columns:
                        ultima_dada.loc[:, 'Time_fin_ms'] = temps_referencia_df['Time_fin_ms'].values[:n]

                    ultima_dada['real'] = False
                #self.data_buffers[algo_name].append(ultima_dada)
                    ts_col = None
                    for c in ['Timestamp', 'time_ms', 'Time_ini_ms', 'Time_fin_ms']:
                        if c in ultima_dada.columns:
                            ts_col = c
                            break

                    if ts_col is not None: # Comprovar que el nuevo bloque no duplica los limites del ultimo bloque del buffer
                        prev = self.data_buffers[algo_name][-1]
                        prev_ts_col = None

                        for c in ['Timestamp', 'time_ms', 'Time_ini_ms', 'Time_fin_ms']:
                            if c in prev.columns:
                                prev_ts_col = c
                                break
                        
                        if prev_ts_col:
                            try:
                                prev_first = int(prev[prev_ts_col].iloc[0])
                                prev_last  = int(prev[prev_ts_col].iloc[-1])
                                new_first  = int(ultima_dada[ts_col].iloc[0])
                                new_last   = int(ultima_dada[ts_col].iloc[-1])

                                if (new_last > prev_last) or (new_first > prev_first):
                                    self.data_buffers[algo_name].append(ultima_dada)
                            
                            except Exception:
                                pass


        global_x_min = None
        global_x_max = None

        current_config = [sv.get() for sv in self.current_selections]
        for i, (algo_name, ax) in enumerate(zip(current_config, self.axes)):
            
            ax.clear()
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.margins(x=0)

            if algo_name == self.DEFAULT_OPTION:
                ax.set_title(f"Grafica {i+1}: Sin selección")
                ax.set_facecolor('#f0f0f0') 
                continue
            
            ax.set_facecolor('white')
            lista_bloques = list(self.data_buffers[algo_name])

            target_col = None
            for idx, bloque in enumerate(lista_bloques):
                val_cols =  [c for c in bloque.columns if c not in ['Timestamp', 'Time_fin_ms', 'Time_ini_ms', 'real']]
                if not val_cols: continue

                if target_col is None:
                    target_col = val_cols[0]

                #x_vals = [epoch1700_to_datetime(t/1000) for t in bloque['Timestamp']]
                ts_col = None
                for c in ['Timestamp', 'time_ms', 'Time_ini_ms', 'Time_fin_ms']:
                    if c in bloque.columns:
                        ts_col = c
                        break

                if ts_col is None:
                    continue
                x_vals = [epoch1700_to_datetime(t/1000) for t in bloque[ts_col]]
                y_vals = bloque[target_col].values

                es_real = bloque.get('real', pd.Series([True] * len(bloque))).iloc[0]
                color_linea = 'tab:blue' if es_real else 'red'

                ax.plot(x_vals, y_vals, color=color_linea, linewidth=1.8)
                if target_col is not None:
                    ax.set_title(f"{algo_name} - {target_col}", fontsize=11, fontweight='bold')
                else:
                    ax.set_title(f"{algo_name} - value", fontsize=11, fontweight='bold')

                if idx < len(lista_bloques) - 1:
                    seguent_bloque = lista_bloques[idx + 1]

                    dt_fin_actual = x_vals[-1]

                    ts_col_next = None
                    for c in ['Timestamp', 'time_ms', 'Time_ini_ms', 'Time_fin_ms']:
                        if c in seguent_bloque.columns:
                            ts_col_next = c
                            break

                    if ts_col_next is None:
                        continue

                    dt_inicio_siguiente = epoch1700_to_datetime(seguent_bloque[ts_col_next].iloc[0]/1000)
                    val_inicio_siguiente = seguent_bloque[target_col].iloc[0]

                    x_conn = [dt_fin_actual, dt_inicio_siguiente]
                    y_conn = [y_vals[-1], val_inicio_siguiente]

                    gap = (dt_inicio_siguiente - dt_fin_actual).total_seconds()

                    if gap > 2.0:
                        # Pintar vermell (error o salt de bloque)
                        ax.plot(x_conn, y_conn, color='red', linewidth=1.8, linestyle='--')
                    else:
                        # Pintar blau (normal)
                        ax.plot(x_conn, y_conn, color=color_linea, linewidth=1.8)

                if global_x_min is None or x_vals[0] < global_x_min: global_x_min = x_vals[0]
                if global_x_max is None or x_vals[-1] > global_x_max: global_x_max = x_vals[-1]
            
            if lista_bloques and target_col is not None:
                ultimo_bloque = lista_bloques[-1]

                ts_col_last = None
                for c in ['Timestamp', 'time_ms', 'Time_ini_ms', 'Time_fin_ms']:
                    if c in ultimo_bloque.columns:
                        ts_col_last = c
                        break

                if ts_col_last is None:
                    continue

                last_x = epoch1700_to_datetime(ultimo_bloque[ts_col_last].iloc[-1]/1000)
                last_y = ultimo_bloque[target_col].iloc[-1]
                ax.plot(last_x, last_y, marker='o', color='red', markersize=6)
                ax.annotate(f'{last_y:.2f}', xy=(last_x, last_y), xytext=(8, 0), textcoords='offset points', color='red', fontsize=15, weight='bold')
        
        if global_x_min and global_x_max:
            self.axes[3].set_xlim(global_x_min, global_x_max)
        
        self.axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def on_plot_click(self, event):
        if event.inaxes is None:
            return
        
        for i, ax in enumerate(self.axes):
            if event.inaxes == ax:
                algo_name = self.current_selections[i].get()
                if algo_name != self.DEFAULT_OPTION:
                    self.open_historical_window(algo_name)
                break

    def open_historical_window(self, algo_name):
        new_win = tk.Toplevel(self)
        new_win.title(f"Historico completo: {algo_name}")
        new_win.geometry("1200x600")

        if not hasattr(self, "historic_last_pred"):
            self.historic_last_pred = {}

        try:
            root = zarr.open(STORE_PATH, mode='r')

            group_path = f"predictions/{algo_name}"
            if group_path not in root:
                print(f"No se encontraron datos históricos para {algo_name}")
                return
            
            subgroups = list(root[group_path].group_keys())
            if not subgroups:
                print(f"No hay subpredicciones en {group_path}")
                return
            
            top = tk.Frame(new_win)
            top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

            tk.Label(top, text="Sub-predicción:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 6))

            cbo_var = tk.StringVar()
            cbo = ttk.Combobox(top, textvariable=cbo_var, values=subgroups, state="readonly", width=30)

            default_pred = self.historic_last_pred.get(algo_name, subgroups[0])
            if default_pred not in subgroups:
                default_pred = subgroups[0]
            
            cbo_var.set(default_pred)
            cbo.pack(side=tk.LEFT, padx=(0, 12))

            def exportar_csv():
                try:
                    import pandas as pd
                    from tkinter import filedialog, messagebox

                    pred_name = cbo_var.get()
                    data_group = root[f"{group_path}/{pred_name}"]

                    if "value" not in data_group:
                        messagebox.showerror("Export CSV", f"Falta 'value' en {group_path}/{pred_name}")
                        return
                    
                    vals = data_group["value"][:]
                    if len(vals) == 0:
                        messagebox.showinfo("Exportar CSV", "No hay datos para exportar en la sub‑predicción seleccionada.")
                        return

                    ts_name = None
                    for c in ["time_ms", "time_fin_ms", "time_ini_ms"]:
                        if c in data_group:
                            ts_name = c
                            break

                    if ts_name is None:
                        messagebox.showerror("Exportar CSV", f"No se encontró serie temporal ('time_ms'/'time_fin_ms'/'time_ini_ms') en {group_path}/{pred_name}")
                        return
                    
                    t_ms = data_group[ts_name][:]
                    n = min(len(t_ms), len(vals))
                    if n <= 0:
                        messagebox.showinfo("Exportar CSV", "No hay muestras temporales para exportar")
                        return
                    
                    df = pd.DataFrame({ts_name: t_ms[:n], "value": vals[:n]})

                    carpeta = filedialog.askdirectory(title="Selecciona la carpeta donde guardar el CSV")
                    if not carpeta:
                        return
                    
                    safe_algo = algo_name.replace(" ", "_")
                    safe_pred = pred_name.replace(" ", "_")
                    nombre = f"{safe_algo}_{safe_pred}.csv"

                    ruta = os.path.join(carpeta, nombre)

                    df.to_csv(ruta, index=False)
                    messagebox.showinfo("Exportar CSV", f"Archivo guardado:\n{ruta}")

                except Exception as e:
                    try:
                        from tkinter import messagebox
                        messagebox.showerror("Export CSV", f"Error al exportar: {e}")
                    
                    except Exception:
                        print(f"[ERROR] Exportar CSV: {e}")

            spacer = tk.Frame(top)
            spacer.pack(side=tk.LEFT, expand=True, fill=tk.X)
            btn_export = ttk.Button(top, text="Export CSV", command=exportar_csv)
            btn_export.pack(side=tk.RIGHT)

            fig_hist, ax_hist = plt.subplots(figsize=(10.5, 5.3))
            ax_hist.margins(x=0)
            ax_hist.grid(True, linestyle='--', alpha=0.5)
            canvas_hist = FigureCanvasTkAgg(fig_hist, master=new_win)
            canvas_hist.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            GAP_SEC = 2.0

            def render_pred(pred_name: str):
                try:
                    import numpy as np
                    ax_hist.clear()
                    ax_hist.margins(x=0)
                    ax_hist.grid(True, linestyle='--', alpha=0.5)

                    def escrivir_en_pantalla (text: str):
                        ax_hist.text(0.5, 0.5, text, transform=ax_hist.transAxes, ha="center", va="center", color="red")
                        fig_hist.canvas.draw_idle()
                    
                    data_group = root[f"{group_path}/{pred_name}"]

                    if "value" not in data_group:
                        escrivir_en_pantalla("Falta 'value' en la sub-predicción {group_path}/{pred_name}")
                        return
                    
                    vals = data_group["value"][:]
                    if len(vals) == 0:
                        escrivir_en_pantalla(f"Sin datos para representar {group_path}/{pred_name}")
                        return
                    
                    ts_name = None
                    if "time_ms" in data_group:
                        ts_name = "time_ms"
                    elif "time_fin_ms" in data_group:
                        ts_name = "time_fin_ms"
                    elif "time_ini_ms" in data_group:
                        ts_name = "time_ini_ms"
                    
                    if ts_name is None:
                        escrivir_en_pantalla(f"No se encontró ninguna serie temporal ('time_ms', 'time_fin_ms' ni 'time_ini_ms') en {group_path}/{pred_name}")
                        return

                    t_ms = data_group[ts_name][:]
                    n = min(len(t_ms), len(vals))

                    if n == 0:
                        escrivir_en_pantalla(f"No hay muestras temporales para representar {group_path}/{pred_name}")
                        return
                    
                    x_dt = [epoch1700_to_datetime(ts/1000) for ts in t_ms[:n]]
                    x_num = mdates.date2num(x_dt)
                    y = vals[:n]

                    #fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
                    #ax_hist.margins(x=0)
                    #ax_hist.grid(True, linestyle='--', alpha=0.5)

                    ax_hist.plot(x_num, y, color='tab:blue', linewidth=1.2)
                    
                    is_rsa = (algo_name.strip().upper() == "RSA")

                    if not is_rsa:
                        dx_days = np.diff(x_num)
                        if dx_days.size > 0:
                            gaps_sec = dx_days * 86400.0
                            idx_gaps = np.where(gaps_sec >= GAP_SEC)[0]
                            for i in idx_gaps:
                                ax_hist.plot([x_num[i], x_num[i+1]], [y[i], y[i+1]], color='red', linestyle='--', linewidth=1.8)

                    ax_hist.set_xlim(x_num[0], x_num[-1])
                    ax_hist.xaxis_date()
                    ax_hist.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    ax_hist.set_title(f"Histórico (Scroll: Zoom | Arrastrar: Mover) - {algo_name} / {pred_name}", fontweight='bold')

                    #ax_hist.grid(True, linestyle='--', alpha=0.5)

                    fig_hist.tight_layout()
                    canvas_hist.draw_idle()

                except Exception as e:
                    print(f"Error al abrir historico interactivo: {e}")

            render_pred(default_pred)

            # canvas_hist = FigureCanvasTkAgg(fig_hist, master=new_win)
            # canvas_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            handler = ZoomPanHandler(ax_hist)
            new_win.handler = handler
            
            fig_hist.canvas.mpl_connect('scroll_event', handler.zoom)
            fig_hist.canvas.mpl_connect('button_press_event', handler.on_press)
            fig_hist.canvas.mpl_connect('button_release_event', handler.on_release)
            fig_hist.canvas.mpl_connect('motion_notify_event', handler.on_motion)

            def on_change_subpred(event=None):
                sel = cbo_var.get()
                self.historic_last_pred[algo_name] = sel
                render_pred(sel)

            cbo.bind("<<ComboboxSelected>>", on_change_subpred)
        
        except Exception as e:
            print(f"Error en el open_historical_window: {e}")

    def on_close(self):
        print("- Cerrando interfaz grafica... ")
        self.stop_flag = True
        try:
            plt.close('all')
            self.quit()
            self.destroy()
        except Exception as e:
            print(f"- Error al cerrar: {e}")

class ZoomPanHandler:
    def __init__(self, ax):
        self.ax = ax
        self.press = None

    def zoom(self, event):
        if event.inaxes != self.ax: return
        base_scale = 1.5
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1

        cur_xlim = self.ax.get_xlim()
        xdata = event.xdata 
        
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        rel_x = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        
        self.ax.set_xlim([xdata - new_width * (1 - rel_x), xdata + new_width * rel_x])
        self.ax.figure.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax: return
        self.press = event.xdata

    def on_release(self, event):
        self.press = None
        self.ax.figure.canvas.draw_idle()

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ax: return
        
        dx = event.xdata - self.press
        cur_xlim = self.ax.get_xlim()
        self.ax.set_xlim(cur_xlim[0] - dx, cur_xlim[1] - dx)
        self.ax.figure.canvas.draw_idle()

if __name__ == "__main__":
    app = RealTimeApp(["Shock Index", "Heart Rate Variability", "RSA"], "SESION_TEST_01")
    app.mainloop()
