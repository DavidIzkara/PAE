import matplotlib
matplotlib.use('TkAgg')

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class RealTimeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Visualización en Tiempo Real")
        self.geometry("1650x850")

        selector_frame = tk.Frame(self)
        selector_frame.pack(pady=1)

        self.variable_options = ['Variable 1', 'Variable 2',
                                 'Variable 3', 'Variable 4']
        self.selected_vars = []
        for i in range(4):
            label = ttk.Label(selector_frame,
                              text=f"Variable gráfica {i+1}:")
            label.grid(row=0, column=i, padx=2)

            combo = ttk.Combobox(selector_frame,
                                 values=self.variable_options,
                                 state="readonly", width=14)
            combo.grid(row=1, column=i, padx=2)
            combo.current(i % len(self.variable_options))
            self.selected_vars.append(combo)

        for combo in self.selected_vars:
            combo.bind("<<ComboboxSelected>>",
                       lambda e: self.draw_plots())

        plot_frame = tk.Frame(self)
        plot_frame.pack(fill="both", expand=True)

        self.fig, self.axs = plt.subplots(
            2, 2, figsize=(12, 6.5), constrained_layout=True
        )
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.canvas.mpl_connect("button_press_event",
                                self.on_graph_click)

        self._stop_flag = False
        self.draw_plots()
        self.refresh_loop()

    def draw_plots(self):
        for i in range(2):
            for j in range(2):
                ax = self.axs[i, j]
                ax.clear()
                var_name = self.selected_vars[i * 2 + j].get()

                x = [1, 2, 3, 4, 5]
                y = [(i + 1) * (j + 1) * val for val in range(1, 6)]

                ax.plot(x, y)
                ax.set_title(var_name)

        self.canvas.draw()

    def refresh_loop(self):
        if not self._stop_flag:
            self.draw_plots()
            self.after(1000, self.refresh_loop)  # cada segundo

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
        var_name = self.selected_vars[idx[0] * 2 + idx[1]].get()

        x = [1, 2, 3, 4, 5]
        y = [(idx[0] + 1) * (idx[1] + 1) * val for val in range(1, 6)]

        ax.plot(x, y)
        ax.set_title(var_name)

        canvas = FigureCanvasTkAgg(fig, master=new_win)
        canvas.get_tk_widget().pack()
        canvas.draw()
