import tkinter as tk
from tkinter import ttk
from threading import Thread
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageTk
from functions_for_run import generate_randomseed
from run_save import run_save_main41_txt, runtxt_to_csv, add_typed_in_data, add_whether_in_the_detector, add_whether_in_the_detector_without_angle
import os
from plotting import plot_llp_decay_in_the_detector
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
matplotlib.use('Agg')
dictionay_for_temp_use = str('')
fig_path_okk = str('')

class App:
    file_path_combined = str('')
    date_csv = str('')
    file_path_combined_detected = str('')
    
    def __init__(self, root):
        self.root = root
        self.root.title("BtoKa.cc LLP parameter setting")
        
        # 创建选择界面
        self.create_choice_interface()
    
    def clear_widgets(self):
        for widget in self.root.winfo_children():
            widget.destroy()
            
    def task_completed(self):
        label = tk.Label(self.root, text = "Task Completed")
        label.pack(pady = 10)

    # def print_dirctionay(self):
    #     label = tk.Label(self.root, text = dictionay_for_temp_use)
    #     label.pack(pady = 10)
    
    def create_choice_interface(self):
        self.clear_widgets()
        label = tk.Label(self.root, text="Choose the Program You'd Like to Run:")
        label.pack(pady=10)

        button_a = tk.Button(self.root, text="Start loop with mass and Br", command=self.run_program_a)
        button_a.pack(pady=5)

        button_b = tk.Button(self.root, text="Start loop with Br and Ctau", command=self.run_program_b)
        button_b.pack(pady=5)
        
        button_c = tk.Button(self.root, text = "Start Analyse LLP Decay Position Roughly", command=self.run_program_judge_roughly)
        button_c.pack(pady=5)
        
        button_c = tk.Button(self.root, text = "Start Analyse LLP Decay Position Precisely", command=self.run_program_judge_precisely)
        button_c.pack(pady=5)
         
        button_d = tk.Button(self.root, text = "Combine Files Roughly", command=self.run_program_combine)
        button_d.pack(pady = 5)

        button_d = tk.Button(self.root, text = "Combine Files Precise", command=self.run_program_combine_precise)
        button_d.pack(pady = 5)
                
        # button_d = tk.Button(self.root, text = "Plot br-ctau fig", command=self.run_program_plot)
        # button_d.pack(pady = 5)



    def run_program_a(self):
        # 清除选择界面
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # 执行已有的程序
        self.setup_program_Br_mass()

    def run_program_b(self):

        for widget in self.root.winfo_children():
            widget.destroy()

        self.setup_program_Br_ctau()
        
    def run_program_judge_precisely(self):
        self.clear_widgets()
        self.set_up_program_judge_precisely()        
        
    def run_program_judge_roughly(self):
        self.clear_widgets()
        self.set_up_program_judge_roughly()
        
    # def run_program_plot(self):
    #     self.clear_widgets()
    #     self.set_up_program_plot_ctau_br()
        
        
    def run_program_combine(self):
        self.clear_widgets()
        self.set_up_program_combine_files()
        
    def run_program_combine_precise(self):
        self.clear_widgets()
        self.set_up_program_combine_files_precise()
        
        


    def setup_program_Br_mass(self):
        style = ttk.Style()
        style.configure("custom.Horizontal.TProgressbar", thickness=200)
        self.progress = ttk.Progressbar(self.root, length=200, style="custom.Horizontal.TProgressbar")
        self.progress.pack(side=tk.BOTTOM)
        self.root.title = 'Br-Mass Simulation'
        # Add entries for all the parameters
        self.pythia8_example_path = self.create_entry("The Path of Your Program", "/Users/shiyuzhe/Documents/University/LLP/Second_Term/pythia8/examples/")
        self.output_path = self.create_entry("The Output Folder Path", "/Users/shiyuzhe/Documents/University/LLP/Second_Term/pythia8/BtoKa/auto_data/Mass_Br/")
        self.mass_lower_lim_entry = self.create_entry("Mass lower limit 10^x")
        self.mass_upper_lim_entry = self.create_entry("Mass upper limit 10^x")
        self.mass_array_length_entry = self.create_entry("Mass array length")
        self.br_lower_lim_entry = self.create_entry("BR lower limit 10^x")
        self.br_upper_lim_entry = self.create_entry("BR upper limit 10^x")
        self.br_array_length_entry = self.create_entry("BR array length")
        self.seed_array_length_entry = self.create_entry("Seed array length")
        self.tau_entry = self.create_entry("Tau", "999")  # Default value is 999

        self.button = tk.Button(self.root, text="Start", command=self.start_thread_mass_br)
        self.button.pack()

        self.back_button = tk.Button(self.root, text="Back to Simulation Type Choice", command=self.create_choice_interface)
        self.back_button.pack(pady=10)
        
    def setup_program_Br_ctau(self):
        style = ttk.Style()
        style.configure("custom.Horizontal.TProgressbar", thickness=200)
        self.progress = ttk.Progressbar(self.root, length=200, style="custom.Horizontal.TProgressbar")
        self.progress.pack(side=tk.BOTTOM)
        self.root.title = 'Br-Ctau Simulation'
        # Add entries for all the parameters
        self.pythia8_example_path = self.create_entry("The Path of Your Program", "/Users/shiyuzhe/Documents/University/LLP/Second_Term/pythia8/examples/")
        self.output_path = self.create_entry("The Output Folder Path", "/Users/shiyuzhe/Documents/University/LLP/Second_Term/pythia8/BtoKa/auto_data/Ctau_Br/")
        self.ctau_lower_lim_entry = self.create_entry("Ctau lower limit 10^x")
        self.ctau_upper_lim_entry = self.create_entry("Ctau upper limit 10^x")
        self.ctau_array_length_entry = self.create_entry("Ctau array length")
        self.br_lower_lim_entry = self.create_entry("BR lower limit 10^x")
        self.br_upper_lim_entry = self.create_entry("BR upper limit 10^x")
        self.br_array_length_entry = self.create_entry("BR array length")
        self.seed_array_length_entry = self.create_entry("Seed array length")
        self.mass_entry = self.create_entry("Mass", "1")  # Default value is 1

        self.button = tk.Button(self.root, text="Start", command=self.start_thread_ctau_br)
        self.button.pack()

        self.back_button = tk.Button(self.root, text="Back to Simulation Type Choice", command=self.create_choice_interface)
        self.back_button.pack(pady=10)

    def create_entry(self, label, default_value=""):
        container = tk.Frame(self.root)
        container.pack()
        tk.Label(container, text=label).pack(side="left")
        entry = tk.Entry(container)
        entry.insert(0, default_value)
        entry.pack(side="right")
        return entry

    def set_up_program_judge_roughly(self):
        style = ttk.Style()
        style.configure("custom.Horizontal.TProgressbar", thickness=200)
        self.progress = ttk.Progressbar(self.root, length=200, style="custom.Horizontal.TProgressbar")
        self.progress.pack(side=tk.BOTTOM)
        self.root.title = 'Analyse Data'
        
        self.file_path_for_judge_llp_decay_position = self.create_entry("The Path of Your Files Folder", "/Users/shiyuzhe/Documents/University/LLP/Second_Term/pythia8/BtoKa/auto_data/test_files/for_test_original_data/")
        
        self.button = tk.Button(self.root, text = "Start", command=self.start_thread_judge_llp_in_detector_roughly)
        self.button.pack()
        
        self.result_label = ttk.Label(self.root, text = "")
        self.result_label.pack(pady = 10)
        
        self.back_button = tk.Button(self.root, text="Back to Simulation Type Choice", command = self.create_choice_interface)
        self.back_button.pack(pady=10)
     
    def set_up_program_judge_precisely(self):
        style = ttk.Style()
        style.configure("custom.Horizontal.TProgressbar", thickness=200)
        self.progress = ttk.Progressbar(self.root, length=200, style="custom.Horizontal.TProgressbar")
        self.progress.pack(side=tk.BOTTOM)
        self.root.title = 'Analyse Data'
        
        self.file_path_for_judge_llp_decay_position = self.create_entry("The Path of Your Files Folder", "/Users/shiyuzhe/Documents/University/LLP/Second_Term/pythia8/BtoKa/auto_data/test_files/for_test_original_data/")
        
        self.button = tk.Button(self.root, text = "Start", command=self.start_thread_judge_llp_in_detector_precisely)
        self.button.pack()
        
        self.result_label = ttk.Label(self.root, text = "")
        self.result_label.pack(pady = 10)
        
        self.back_button = tk.Button(self.root, text="Back to Simulation Type Choice", command = self.create_choice_interface)
        self.back_button.pack(pady=10)
        
    def set_up_program_combine_files(self):
        style = ttk.Style()
        style.configure("custom.Horizontal.TProgressbar", thickness=200)
        self.progress = ttk.Progressbar(self.root, length=200, style="custom.Horizontal.TProgressbar")
        self.progress.pack(side=tk.BOTTOM)
        self.root.title = 'Combine Files'
        
        self.complete_file_path = self.create_entry("Your Complete file folder path", "/Users/shiyuzhe/Documents/University/LLP/Second_Term/pythia8/BtoKa/auto_data/test_files/Completed_llp_data/")
        
        self.button1 = tk.Button(self.root, text = "Start", command=self.start_thread_combine_files)
        self.button1.pack()
        
        # self.fig_path_label= tk.Label(self.root, text = "The fig Path: ")
        # self.fig_path_label.pack(pady=10)
        
        self.fig_path = ttk.Label(self.root, text = "")
        self.fig_path.pack(pady=10)
        
        # self.button2 = tk.Button(self.root, text = "Show fig", command=self.start_thread_show_image)
        # self.button2.pack()
        
        self.back_button = tk.Button(self.root, text="Back to Simulation Type Choice", command=self.create_choice_interface)
        self.back_button.pack(pady=10)

    def set_up_program_combine_files_precise(self):
        style = ttk.Style()
        style.configure("custom.Horizontal.TProgressbar", thickness=200)
        self.progress = ttk.Progressbar(self.root, length=200, style="custom.Horizontal.TProgressbar")
        self.progress.pack(side=tk.BOTTOM)
        self.root.title = 'Combine Files'
        
        self.complete_file_path = self.create_entry("Your Complete file folder path", "/Users/shiyuzhe/Documents/University/LLP/Second_Term/pythia8/BtoKa/auto_data/test_files/Completed_llp_data_precise/")
        
        self.button1 = tk.Button(self.root, text = "Start", command=self.start_thread_combine_files_precise)
        self.button1.pack()
        
        # self.fig_path_label= tk.Label(self.root, text = "The fig Path: ")
        # self.fig_path_label.pack(pady=10)
        
        self.fig_path = ttk.Label(self.root, text = "")
        self.fig_path.pack(pady=10)
        
        # self.button2 = tk.Button(self.root, text = "Show fig", command=self.start_thread_show_image)
        # self.button2.pack()
        
        self.back_button = tk.Button(self.root, text="Back to Simulation Type Choice", command=self.create_choice_interface)
        self.back_button.pack(pady=10)
      
    def start_thread_mass_br(self):
        Thread(target=self.loop_mass_br).start()
    
    def start_thread_ctau_br(self):
        Thread(target=self.loop_ctau_br).start()
        
    def start_thread_judge_llp_in_detector_precisely(self):
        Thread(target=self.judge_llp_in_detector).start()
        
    def start_thread_judge_llp_in_detector_roughly(self):
        Thread(target=self.judge_llp_in_detector_roughly).start()
        
    def start_thread_plot_fig_br_ctau(self):
        Thread(target=self.plot_llp).start()
        
    def start_thread_combine_files(self):
        Thread(target=self.combine_files).start()
        
    def start_thread_combine_files_precise(self):
        Thread(target=self.combine_files_precise).start()
        
    def start_thread_show_image(self):
        Thread(target=self.show_image).start()  
    

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
                    date_csv_files_path = runtxt_to_csv(run_save_main41_txt(mass, seed, br, tau, out_put_path, main41_path)[0])[0]

                    self.progress["value"] += 1
                    self.root.update()
        
        # print('Task Completed')
        self.date_csv = os.path.dirname(date_csv_files_path)
        # print(self.date_csv)
        self.root.after(0, self.task_completed)
    
    
    def loop_ctau_br(self):
        ctau_lower_lim = float(self.ctau_lower_lim_entry.get())
        ctau_upper_lim = float(self.ctau_upper_lim_entry.get())
        ctau_array_length = int(self.ctau_array_length_entry.get())
        br_lower_lim = float(self.br_lower_lim_entry.get())
        br_upper_lim = float(self.br_upper_lim_entry.get())
        br_array_length = int(self.br_array_length_entry.get())
        seed_array_length = int(self.seed_array_length_entry.get())
        out_put_path = str(self.output_path.get())
        main41_path = str(self.pythia8_example_path.get())
        mass = str(self.mass_entry.get())
        total_iterations = seed_array_length * br_array_length * ctau_array_length
        self.progress["maximum"] = total_iterations
        
        with tqdm(total = total_iterations) as pbar:
            for seed in generate_randomseed(seed_array_length):
                for br in np.logspace(br_lower_lim, br_upper_lim, br_array_length):
                    for ctau in np.logspace(ctau_lower_lim, ctau_upper_lim, ctau_array_length):
                        temp1 = run_save_main41_txt(mass, seed, br, ctau, out_put_path, main41_path)
                        date_csv_file_path = runtxt_to_csv(temp1[0])[0]
                        
                        
                        self.progress["value"] += 1
                        self.root.update()
        self.date_csv = os.path.dirname(date_csv_file_path)
        # print(self.date_csv)
        # print('Task Completed')
        self.root.after(0, self.task_completed)
        # self.root.after(0, self.print_dirctionay)   
    
    def judge_llp_in_detector_roughly(self):
        folder_path = str(self.file_path_for_judge_llp_decay_position.get())
        self.progress["maximum"] = len(os.listdir(folder_path))
        for files in os.listdir(folder_path):
            file_path_all = os.path.join(folder_path, files)
            # print(file_path_all)
            if os.path.isfile(file_path_all):  # Ensure we only process file, not dirctionary
                # print(file_path_all)
                # add_typed_in_data(files, folder_path)
                # print(add_whether_in_the_detector_without_angle(file_path_all, folder_path))
                add_whether_in_the_detector_without_angle(file_path_all, folder_path)
            self.progress["value"] += 1
            self.root.update()
            
        self.result_label.config(text = f"{os.path.dirname(os.path.dirname(folder_path))}"+"/Completed_llp_data/")
        return True
    
    def judge_llp_in_detector(self):
        folder_path = str(self.file_path_for_judge_llp_decay_position.get())
        self.progress["maximum"] = len(os.listdir(folder_path))
        for files in os.listdir(folder_path):
            file_path_all = os.path.join(folder_path, files)
            # print(file_path_all)
            if os.path.isfile(file_path_all):  # Ensure we only process file, not dirctionary
                # print(file_path_all)
                # add_typed_in_data(files, folder_path)
                # print(add_whether_in_the_detector(file_path_all, folder_path))
                add_whether_in_the_detector(file_path_all, folder_path)
            self.progress["value"] += 1
            self.root.update()
            
        self.result_label.config(text = f"{os.path.dirname(os.path.dirname(folder_path))}"+"/Completed_llp_data_precise/")
        return True
    
    def combine_files(self):
        merged_df = pd.DataFrame()
        df_all = pd.DataFrame()
        completed_file_path = str(self.complete_file_path.get())
        out_file_path = os.path.dirname(completed_file_path)
        self.progress["maximum"] = len(os.listdir(completed_file_path))
        total_iterations = len(os.listdir(completed_file_path))
        with tqdm(total = total_iterations) as pbar:
            for file in os.listdir(completed_file_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(completed_file_path, file)
                    
                    # with open(file_path, 'rb') as f:
                    #     raw_data = f.read()
                    #     result = chardet.detect(raw_data)
                    #     encoding_code = result['encoding']
                    
                    df = pd.read_csv(file_path)
                    
                    detected_df = df[df['detected'] == 1]
                    merged_df = pd.concat([merged_df, detected_df], ignore_index=True)
                    df_all = pd.concat(([df_all, df]), ignore_index=True)
                    
                    # print(file + 'has been combined')
                self.progress["value"] += 1
                self.root.update()
            self.file_path_combined_detected = os.path.dirname(out_file_path) + '/detected_combined_file.csv'
            self.file_path_combined = os.path.dirname(out_file_path) + '/all_combined_file.csv'
            merged_df.to_csv(self.file_path_combined_detected)
            df_all.to_csv(self.file_path_combined)
        # plt.title('Scatter Plot of Detected Data')
        # plt.xlabel('br[B->K LLP]')
        # plt.ylabel('tau[cm]')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        # fig_path = os.path.join(completed_file_path, 'Br_ctau_fig.png')
        # plt.savefig(fig_path)
        # plt.close()
        # fig_path_okk = fig_path
        # self.fig_path.config(text = fig_path_okk)
        return self.file_path_combined, self.file_path_combined_detected

    def combine_files_precise(self):
        merged_df = pd.DataFrame()
        df_all = pd.DataFrame()
        completed_file_path = str(self.complete_file_path.get())
        out_file_path = os.path.dirname(completed_file_path)
        self.progress["maximum"] = len(os.listdir(completed_file_path))
        total_iterations = len(os.listdir(completed_file_path))
        with tqdm(total = total_iterations) as pbar:
            for file in os.listdir(completed_file_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(completed_file_path, file)
                    
                    # with open(file_path, 'rb') as f:
                    #     raw_data = f.read()
                    #     result = chardet.detect(raw_data)
                    #     encoding_code = result['encoding']
                    
                    df = pd.read_csv(file_path)
                    
                    detected_df = df[df['detected'] == 1]
                    merged_df = pd.concat([merged_df, detected_df], ignore_index=True)
                    df_all = pd.concat(([df_all, df]), ignore_index=True)
                    
                    # print(file + 'has been combined')
                self.progress["value"] += 1
                self.root.update()
            self.file_path_combined_detected = os.path.dirname(out_file_path) + '/detected_combined_precise_file.csv'
            self.file_path_combined = os.path.dirname(out_file_path) + '/all_combined_precise_file.csv'
            merged_df.to_csv(self.file_path_combined_detected)
            df_all.to_csv(self.file_path_combined)
        # plt.title('Scatter Plot of Detected Data')
        # plt.xlabel('br[B->K LLP]')
        # plt.ylabel('tau[cm]')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        # fig_path = os.path.join(completed_file_path, 'Br_ctau_fig.png')
        # plt.savefig(fig_path)
        # plt.close()
        # fig_path_okk = fig_path
        # self.fig_path.config(text = fig_path_okk)
        return self.file_path_combined, self.file_path_combined_detected

    def plot_completed_csv(self):
        df = pd.read_csv(self.file_path_combined)
        plt.title('Scatter Plot of Detected Data')
        plt.xlabel('br[B->K LLP]')
        plt.ylabel('tau[cm]')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        fig_path = os.path.join(os.path.dirname(self.completed_file_path), '/Br_ctau_fig.png')
        plt.savefig(fig_path)
        plt.close()
        fig_path_okk = fig_path
        self.fig_path.config(text = fig_path_okk)
        

    def show_image(self):
        new_window = tk.Toplevel()
        
        image_path = self.fig_path.cget("text")
        print(image_path)
        
        try:
            img = Image.open(image_path)
            
            # 把PIL图像对象转换为Tkinter可用的PhotoImage对象
            imgtk = ImageTk.PhotoImage(img)
            
            # 创建一个标签，使用图片作为内容
            label_img = tk.Label(new_window, image=imgtk)
            label_img.image = imgtk  # 注意：需要保持对PhotoImage对象的引用
            
            # 将标签添加到窗口并显示
            label_img.pack()
        except Exception as e:
            print(f"Error opening image: {e}")


root = tk.Tk()
app = App(root)
# app.show_image()
root.mainloop()

