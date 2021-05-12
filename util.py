import csv, pandas as pd, numpy as np
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter import Tk

def get_csv():
    Tk().withdraw()
    file_directory = askopenfilename(title = "Select your csv",  filetypes= (("csv files", "*.csv"),("all files", "*.*")))
    data = pd.read_csv(file_directory, delimiter=',', header=None, dtype={'zero_column_name': object})
    return np.array(data)

def write_csv(data):
    data = np.stack(data).T
    Tk().withdraw()
    directory = askdirectory()
    with open(directory + '/output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print("File saved in path: {}".format(directory + '/outputs.csv'))