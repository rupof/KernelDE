import pandas as pd
from circuits.circuits import *
from experiment_inputs import experiment_list_total  # Import the configurations_list
import sys
import multiprocessing 
import os
import time
from solvers.MMR.PQK_solver import PQK_solver 
from solvers.MMR.FQK_solver import FQK_solver
from solvers.MMR.kernel_solver import Solver


from utils.rbf_kernel_tools import *
from scipy.integrate import odeint


start = time.time()

results_path = "./data/results/" #./data/results/ for local, /data/results/ for server
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


print("Number of experiments:", len(experiment_list))
print("Index of experiment list:", index_experiment_list)
print("Number of cores:", num_cores)

#save experiment_list to a file as npy

print("Starting the experiment list")

x_span = np.linspace(0.0001, 1, 40)


cache = {}
result_dic_list = []
for idx, experiment in enumerate(experiment_list):
    print(f"Starting experiment {idx} with the following parameters: {experiment}")
    g = experiment["g"]
    f_initial = experiment["f_initial"]


    if experiment["method"] == "PQK":
        OSolver = PQK_solver(experiment["circuit_information"],
                                experiment["executor_type"], 
                                envelope={"function": rbf_kernel_manual, 
                                            "derivative_function": analytical_derivative_rbf_kernel, 
                                            "sigma": experiment["sigma"]})
        dict_to_save = {"sigma": experiment["sigma"]}

    elif experiment["method"] == "FQK":
        OSolver = FQK_solver(experiment["circuit_information"],
                                experiment["executor_type"])
    elif experiment["method"] == "classical_RBF":
        RBF_kernel_list = [rbf_kernel_manual(x_span, x_span, sigma = experiment["sigma"]), analytical_derivative_rbf_kernel(x_span, x_span, sigma = experiment["sigma"])]
        OSolver = Solver(RBF_kernel_list)
        dict_to_save = {"sigma": experiment["sigma"]}
        
    solution, kernel_list = OSolver.solver(x_span, f_initial, g)
    f_sol = solution[0]
    optimal_alpha = solution[1]

    solution_label = f"{experiment['g_name']}_f_initial"
    if solution_label in cache:
        numerical_solution = cache[solution_label]
    else:
        numerical_solution = odeint(g, f_initial, x_span[:])
        cache[solution_label] = numerical_solution
    
    mse = np.mean((f_sol - cache[solution_label]))**2

    dict_to_save = {"f_sol": f_sol, 
                    "optimal_alpha": optimal_alpha, 
                    "mse": mse}
    
    #include all keys and values from experiment["circuit_information"] to the dict_to_save
    for key, value in experiment["circuit_information"].items():
        if key == "encoding_circuit":
            try:
                dict_to_save["CI_encoding_circuit_label"] = value.__name__
            except:
                dict_to_save["CI_encoding_circuit_label"] = value
        elif key == "executor":
            try:
                dict_to_save["CI_executor_type_label"] = value.__name__
            except:
                dict_to_save["CI_executor_type_label"] = value
        elif key == "g":
            pass
        else:
            dict_to_save["CI_"+ key] = value
    for key, value in experiment.items():
        if key == "circuit_information":
            pass
        elif key == "g":
            pass
        elif key == "executor_type":
            pass
        else:
            dict_to_save[key] = value
    print(dict_to_save)
    result_dic_list.append(pd.DataFrame([dict_to_save]))


    df = pd.concat(result_dic_list, axis=0, sort=False, ignore_index=True)
    df.reset_index(drop=True, inplace=True)

    path = results_folder_path + f"/{idx}_T.feather"
    df.to_feather(path)
    print(f"Experiment {idx} finished")
    print("Saved at:", path)

    