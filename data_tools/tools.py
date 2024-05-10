import numpy as np
import pandas as pd
import h5py
import glob
import time 


from pathlib import Path
utils_folder = Path(__file__).parent

#make the file importable from the root folder
import sys
sys.path.append(str(utils_folder.parent))


def load_feather_folder_as_pd(folder_with_temp_files, short_load = False):
    """
    Load a folder or a list of feather file paths as a pandas dataframe

    """
    #above line as a loop
    if isinstance(folder_with_temp_files, str):
        import glob
        temp_files = glob.glob(folder_with_temp_files + "/*.feather")
    elif isinstance(folder_with_temp_files, list):
        temp_files = folder_with_temp_files
    dicts = []
    times = []
    zero_time = time.time()
    if short_load:
        temp_files = temp_files[:200]
    for idx, temp_file in enumerate(temp_files):    
        dict_idx = pd.read_feather(temp_file)
        
        #only grab experiment of row idx
        dict_idx = dict_idx.iloc[idx]
        #open the log file and record the gradient and loss history
        dict_loss = pd.read_csv(temp_file[:-8]+".log", delim_whitespace=True)

        dict_idx["loss_history"] = np.array(dict_loss["f(x)"])
        dict_idx["gradient_history"] = np.array(dict_loss["Gradient"])
        dicts.append(dict_idx)
        times.append(time.time()-zero_time)
    df = pd.concat(dicts, axis=1, sort=False, ignore_index=True)
    print(time.time()-zero_time)
    return df.T

def load_log_data(filename, short_load = False):
    """
    Load a log file as a pandas dataframe
    """
    #above line as a loop
    
        #open the log file and record the gradient and loss history
    dict_loss = pd.read_csv(filename, delim_whitespace=True)
    
    return df.T

def load_log_data(log_file_path):
    """
    Load a log file as a pandas dataframe

    """
    with open(log_file_path, "r") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(eval(line))
    df = pd.DataFrame(data)
    return df