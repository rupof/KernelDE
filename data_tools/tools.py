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
        dicts.append(pd.read_feather(temp_file))
        times.append(time.time()-zero_time)
        #print(temp_file)
    df = pd.concat(dicts, axis=0, sort=False, ignore_index=True)
    print(time.time()-zero_time)
    return df