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
from squlearn.qnn.loss import ODELoss
from squlearn.optimizers import SLSQP, Adam
from squlearn.qnn import QNNRegressor
from squlearn.observables import *


from utils.rbf_kernel_tools import analytical_derivative_rbf_kernel, analytical_derivative_rbf_kernel_2, rbf_kernel_manual
from scipy.integrate import odeint


start = time.time()

results_path = "./data/results/" #./data/results/ for local, /datax/results/ for server
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



cache = {}
result_dic_list = []
for idx, experiment in enumerate(experiment_list):
    print(f"Starting experiment {idx} with the following parameters: {experiment}")
    x_span = experiment["x_domain"]
    loss = experiment["loss"]
    grad_loss = experiment["grad_loss"]
    f_initial = experiment["f_initial"]
    quantum_bandwidth = experiment["quantum_bandwidth"]
    x_span *= quantum_bandwidth


    if experiment["method"] == "PQK":
        OSolver = PQK_solver(experiment["circuit_information"],
                                experiment["executor_type"], 
                                envelope={"function": rbf_kernel_manual, 
                                            "derivative_function": analytical_derivative_rbf_kernel, 
                                            "second_derivative_function": analytical_derivative_rbf_kernel_2,
                                            "sigma": experiment["sigma"]})
        dict_to_save = {"sigma": experiment["sigma"]}

    elif experiment["method"] == "FQK":
        OSolver = FQK_solver(experiment["circuit_information"],
                                experiment["executor_type"])
    elif experiment["method"] == "classical_RBF":
        RBF_kernel_list = [rbf_kernel_manual(x_span, x_span, sigma = experiment["sigma"]), 
                           analytical_derivative_rbf_kernel(x_span, x_span, sigma = experiment["sigma"]),
                           analytical_derivative_rbf_kernel_2(x_span, x_span, sigma = experiment["sigma"])]
        OSolver = Solver(RBF_kernel_list)
        dict_to_save = {"sigma": experiment["sigma"]}
    elif experiment["method"].startswith("QNN"):
        method, boundary_handling = experiment["method"].split("_")
        loss_ODE = ODELoss(loss, grad_loss, initial_vec = f_initial, eta=1, boundary_handling = boundary_handling)
        Optimizer = Adam(options={"maxiter": 400, "tol": 0.00009,  "log_file": results_folder_path + f"/{idx}_T.log"})
        EncodingCircuit = experiment["circuit_information"]["encoding_circuit"]
        #pop the encoding_circuit from the dict
        experiment["circuit_information"].pop("encoding_circuit")
        encoding_circuit = EncodingCircuit(num_features = 1,  **experiment["circuit_information"])
        num_qubits = experiment["circuit_information"]["num_qubits"]
        Observables = SummedPaulis(num_qubits, include_identity=False)                                                      
        param_ini = encoding_circuit.generate_initial_parameters(seed=1)
        param_obs = Observables.generate_initial_parameters(seed=1)
         #np.ones(num_qubits+1)

        clf = QNNRegressor(
            encoding_circuit,
            Observables,
            experiment["executor_type"],
            loss_ODE,
            Optimizer,
            param_ini,
            param_obs,
            opt_param_op = False
        )    

        y_ODE = np.zeros((x_span.shape[0]))
        clf._fit(x_span, y_ODE,  weights=None)
        y_pred = clf.predict(x_span)
        params = clf._param
    
    if experiment["method"].startswith("QNN") == False:    
        solution, kernel_list = OSolver.solver(x_span, f_initial, loss)
        f_sol = solution[0]
        optimal_alpha = solution[1]
    else:
        f_sol = y_pred
        print(params)
        optimal_alpha = params


    solution_label = f"{experiment['loss_name']}_f_initial"
    if solution_label in cache:
        numerical_solution = cache[solution_label]
    else:
        print("Experiment ", experiment["loss_name"], " with f_initial")
        numerical_solution = odeint(experiment["derivatives_of_loss"], f_initial, x_span[:])
        cache[solution_label] = numerical_solution
    
    mse = np.mean((f_sol - cache[solution_label][:,0]))**2

    dict_to_save = {"f_sol": f_sol, 
                    "optimal_alpha": optimal_alpha, 
                    "mse": mse, 
                    "method": experiment["method"],
                    "loss_name": experiment["loss_name"],
                    "domain": experiment["x_domain"]}
    
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
        else:
            dict_to_save["CI_"+ key] = value
    for key, value in experiment.items():
        if key == "circuit_information":
            pass
        elif key == "loss" or "derivatives_of_loss" or "grad_loss":
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

    