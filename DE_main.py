import pandas as pd
from circuits.circuits import *
from experiment_inputs import experiment_list_total  # Import the configurations_list
import sys
import multiprocessing 
import os
import time
from DE_Library.qnn_and_kernels_wrappers import ODELoss_wrapper, executor_type_dictionary
from solvers.wrapper_solver import wrapper_experiment_solver

from utils.rbf_kernel_tools import analytical_derivative_rbf_kernel, analytical_derivative_rbf_kernel_2, rbf_kernel_manual
from scipy.integrate import odeint




start = time.time()


if os.name == 'posix':
    results_path = "/datax/results/"
elif os.name == 'nt':
    results_path = "./data/results/" # #server: /datax/results/

index = str(sys.argv[1])
index_experiment_list = int(sys.argv[2])
num_cores = int(sys.argv[3])

print("Index:", index)
print("Index of experiment list:", index_experiment_list)
print("Number of cores:", num_cores)

experiment_list = experiment_list_total[index_experiment_list]
results_folder_path = results_path + f"DE_{index}_{index_experiment_list}"
if not os.path.exists(results_folder_path):
    os.makedirs(results_folder_path)

for idx, experiment in enumerate(experiment_list):
    results_performance_item_path = os.path.join(results_folder_path, f"{idx}")
    experiment["path"] = results_performance_item_path


print("Number of experiments:", len(experiment_list))
print("Index of experiment list:", index_experiment_list)
print("Number of cores:", num_cores)

#save experiment_list to a file as npy

print("Starting the experiment list")


#experiment_tuple_list = [(experiment_params, experiment_params["results_path"], constants) for experiment_params in df_list_of_dicts]
experiment_tuple_list = [(experiment_params) for experiment_params in experiment_list]
print(len(experiment_tuple_list))

cache = {}
result_dic_list = []
    
if __name__ == "__main__":
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(wrapper_experiment_solver, experiment_tuple_list)
        pool.close()
        pool.join()

        end = time.time()
        print("Performance calculations are done! Now merging the temporary files")
        #merge_temporary_files(results_performance_folder_path, 
        #                      results_path + f"performance_quantum_{index}_{index_experiment_list}.h5", ignore_errors=True)
        print("Everything done!")
        print("Time taken:", end - start)