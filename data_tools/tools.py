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
    print(temp_files)
    for idx, temp_file in enumerate(temp_files):    
        dict_idx = pd.read_feather(temp_file)
        #only grab experiment of row idx
        dict_idx = dict_idx.iloc[0]
        #open the log file and record the gradient and loss history

        try:
            dict_loss = pd.read_csv(temp_file[:-8]+".log", delim_whitespace=True)
            dict_idx["loss_history"] = np.array(dict_loss["f(x)"])
            dict_idx["mse_history"] = np.array(dict_loss["MSE"])
            dict_idx["gradient_history"] = np.array(dict_loss["Gradient"])
        except:
            pass

        dicts.append(dict_idx)
        times.append(time.time()-zero_time)
    df = pd.concat(dicts, axis=1, sort=False, ignore_index=True)
    print(time.time()-zero_time)
    return df.T


def load_log_data(folder_with_temp_files, short_load = False):
    """
    Load a log file as a pandas dataframe
    """
    #above line as a loop
    
        #open the log file and record the gradient and loss history
    if isinstance(folder_with_temp_files, str):
        import glob
        temp_files = glob.glob(folder_with_temp_files + "/*.log")
    elif isinstance(folder_with_temp_files, list):
        temp_files = folder_with_temp_files
    dicts = []
    times = []
    for idx, temp_file in enumerate(temp_files[:]):    
        dict_idx = {"name":temp_file}
        try:
            dict_loss = pd.read_csv(temp_file, delim_whitespace=True)
            dict_idx["loss_history"] = np.array(dict_loss["f(x)"])
            dict_idx["mse_history"] = np.array(dict_loss["MSE"])
            dict_idx["gradient_history"] = np.array(dict_loss["Gradient"])
        except:
            pass
        dicts.append(pd.DataFrame([dict_idx]))
        print(dict_idx)
        times.append(time.time())
    df = pd.concat(dicts, axis=0, sort=False, ignore_index=True)
    print(time.time())
    return df.T



