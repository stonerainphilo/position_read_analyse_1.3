from loop import loop_mass_br
import tkinter as tk
from tkinter import ttk
from threading import Thread
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageTk
from functions_for_run import generate_randomseed
from run_save import run_save_main41_txt, runtxt_to_csv

class App:
    def __init__(self, root):
        self.root = root
        root.title("BtoKa.cc LLP parameter setting")
        style = ttk.Style()
        style.configure("custom.Horizontal.TProgressbar", thickness=200)
        self.progress = ttk.Progressbar(root, length=200, style="custom.Horizontal.TProgressbar")
        self.progress.pack(side = tk.BOTTOM)

        # Add entries for all the parameters
        self.pythia8_example_path = self.create_entry("The Path of Your Program", "/Users/shiyuzhe/Documents/University/LLP/Second_Term/pythia8/examples/")
        self.output_path = self.create_entry("The Output Folder Path", "/Users/shiyuzhe/Documents/University/LLP/Second_Term/pythia8/BtoKa/auto_data/")
        self.mass_lower_lim_entry = self.create_entry("Mass lower limit 10^x")
        self.mass_upper_lim_entry = self.create_entry("Mass upper limit 10^x")
        self.mass_array_length_entry = self.create_entry("Mass array length")
        self.br_lower_lim_entry = self.create_entry("BR lower limit 10^x")
        self.br_upper_lim_entry = self.create_entry("BR upper limit 10^x")
        self.br_array_length_entry = self.create_entry("BR array length")
        self.seed_array_length_entry = self.create_entry("Seed array length")
        self.tau_entry = self.create_entry("Tau", "999")  # Default value is 999

        self.button = tk.Button(root, text="Start", command=self.start_thread)
        self.button.pack()

    def create_entry(self, label, default_value=""):
        container = tk.Frame(self.root)
        container.pack()
        tk.Label(container, text=label).pack(side="left")
        entry = tk.Entry(container)
        entry.insert(0, default_value)
        entry.pack(side="right")
        return entry

    def start_thread(self):
        Thread(target=self.loop_mass_br).start()

    def loop_mass_br(self):
        mass_lower_lim = float(self.mass_lower_lim_entry.get())
        mass_upper_lim = float(self.mass_upper_lim_entry.get())
        mass_array_length = int(self.mass_array_length_entry.get())
        br_lower_lim = float(self.br_lower_lim_entry.get())
        br_upper_lim = float(self.br_upper_lim_entry.get())
        br_array_length = int(self.br_array_length_entry.get())
        seed_array_length = int(self.seed_array_length_entry.get())
        out_put_path = str(self.output_path.get())
        main41_path = str(self.pythia8_example_path.get())
        
        
        tau = float(self.tau_entry.get())

        total_iterations = seed_array_length * mass_array_length * br_array_length

        self.progress["maximum"] = total_iterations

        for seed in generate_randomseed(seed_array_length):
            for mass in np.logspace(mass_lower_lim, mass_upper_lim, mass_array_length):
                for br in np.logspace(br_lower_lim, br_upper_lim, br_array_length):
                    all_name, trimed_name = runtxt_to_csv(run_save_main41_txt(mass, seed, br, tau, out_put_path, main41_path)[0])

                    self.progress["value"] += 1
                    self.root.update()
    
    def show_image(image_path):
    # 创建一个新的tkinter窗口
        new_window = tk.Toplevel()
        
        # 使用PIL库打开图片
        img = Image.open(image_path)
        
        # 把PIL图像对象转换为Tkinter可用的PhotoImage对象
        imgtk = ImageTk.PhotoImage(img)
        
        # 创建一个标签，使用图片作为内容
        label_img = tk.Label(new_window, image=imgtk)
        label_img.image = imgtk  # 注意：需要保持对PhotoImage对象的引用
        
        # 将标签添加到窗口并显示
        label_img.pack()

root = tk.Tk()
app = App(root)
# app.show_image()
root.mainloop()

