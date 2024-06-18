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
from squlearn.optimizers import SLSQP, Adam, SGLBO
from squlearn.qnn import QNNRegressor
from squlearn.observables import *
from DE_Library.qnn_and_kernels_wrappers import ODELoss_wrapper, executor_type_dictionary
from utils.rbf_kernel_tools import analytical_derivative_rbf_kernel, analytical_derivative_rbf_kernel_2, rbf_kernel_manual
from utils.rbf_kernel_tools import matrix_rbf, matrix_rbf_dx_slow, matrix_rbf_dxdx_slow, matrix_rbf_dxdy_slow
from scipy.integrate import odeint



def get_mse_loss(functional_loss_by_iteration, y_exact):
    return np.mean((np.array(functional_loss_by_iteration)-y_exact)**2, axis = 1)
cache = {}

def wrapper_experiment_solver(experiment):
    x_span = experiment["x_domain"]
    loss = experiment["loss"]
    grad_loss = experiment["grad_loss"]
    f_initial = experiment["f_initial"]
    quantum_bandwidth = experiment["quantum_bandwidth"]
    x_span *= quantum_bandwidth
    executor_str = experiment["executor_type"]
    if experiment["circuit_information"]["encoding_circuit"] == "NoCircuit":
        encoding_circuit_label = "NoCircuit"
    else:
        encoding_circuit_label = experiment["circuit_information"]["encoding_circuit"].__name__
    experiment_path = experiment["path"]

    executor_object = executor_type_dictionary[executor_str]

    try:
        if "num_shots" in experiment["method_information"]:
            executor_object.set_shots(experiment["method_information"]["num_shots"])
    except:
        pass

    solution_label = f"{experiment['loss_name']}_f_initial"

    if solution_label in cache:
        numerical_solution = cache[solution_label]
    else:
        print("Experiment ", experiment["loss_name"], " with f_initial")
        numerical_solution = odeint(experiment["derivatives_of_loss"], f_initial, x_span[:])
        cache[solution_label] = numerical_solution

    #print experiment details

    print("Experiment details:")
    print(experiment)

    if experiment["method"] == "PQK":
        OSolver = PQK_solver(experiment["circuit_information"],
                                executor_object, 
                                envelope={"function": matrix_rbf, 
                                            "derivative_function": matrix_rbf_dx_slow, 
                                            "second_derivative_function": matrix_rbf_dxdx_slow,
                                            "mixed_derivative_function": matrix_rbf_dxdy_slow,
                                            "sigma": experiment["sigma"]})
        dict_to_save = {"sigma": experiment["sigma"]}
        experiment["circuit_information"].pop("encoding_circuit")

    elif experiment["method"] == "FQK":
        OSolver = FQK_solver(experiment["circuit_information"],
                                executor_object)
        experiment["circuit_information"].pop("encoding_circuit")

    elif experiment["method"] == "classical_RBF":
        RBF_kernel_list = [rbf_kernel_manual(x_span, x_span, sigma = experiment["sigma"]), 
                           analytical_derivative_rbf_kernel(x_span, x_span, sigma = experiment["sigma"]),
                           analytical_derivative_rbf_kernel_2(x_span, x_span, sigma = experiment["sigma"])]
        OSolver = Solver(RBF_kernel_list)
        dict_to_save = {"sigma": experiment["sigma"]}
    elif experiment["method"] == "QNN":
        method = experiment["method"], 
        method_information_copy = experiment["method_information"].copy()
        boundary_handling = method_information_copy.pop("boundary_handling")
        eta = method_information_copy.pop("eta")
        if "num_shots" in method_information_copy:
            method_information_copy.pop("num_shots")

        if method_information_copy["optimizer"] == "Adam":
            Optimizer = Adam(options={"log_file": experiment["path"] + f".log", **method_information_copy})
        elif method_information_copy["optimizer"] == "SGLBO":
            Optimizer = SGLBO(options={"log_file": experiment["path"] + f".log", **method_information_copy })

        loss_ODE = ODELoss_wrapper(loss, grad_loss, initial_vec = f_initial, eta=eta, boundary_handling = boundary_handling, true_solution=numerical_solution[:,0].flatten())
        EncodingCircuit = experiment["circuit_information"]["encoding_circuit"]
        #pop the encoding_circuit from the dict
        experiment["circuit_information"].pop("encoding_circuit")
        encoding_circuit = EncodingCircuit(num_features = 1,  **experiment["circuit_information"])
        num_qubits = experiment["circuit_information"]["num_qubits"]
        Observables = SummedPaulis(num_qubits, include_identity=False)                                                      
        param_ini = encoding_circuit.generate_initial_parameters(seed=1)
        param_obs = Observables.generate_initial_parameters(seed=1)
        #np.ones(num_qubits+1)
        if executor_str == "pennylane_shots_variance" or executor_str == "qasm_simulator_variance" or executor_str == "qiskit_shots_variance":
            print(f"using variance with {executor_str}")
            variance_for_qnn_regularization = 10**-3
        else:
            variance_for_qnn_regularization = None

        clf = QNNRegressor(
            encoding_circuit,
            Observables,
            executor_object,
            loss_ODE,
            Optimizer,
            param_ini,
            param_obs,
            opt_param_op = False,
            variance_for_qnn_regularization = variance_for_qnn_regularization
        )    

        y_ODE = np.zeros((x_span.shape[0]))
        clf._fit(x_span, y_ODE,  weights=None)
        y_pred = clf.predict(x_span)
        params = clf._param
    
    print(experiment["method"])
    if experiment["method"].startswith("QNN") == False:    
        solution, loss_by_iteration = OSolver.solver(x_span, f_initial, loss)
        f_sol = solution[0]
        optimal_alpha = solution[1]
        ode_loss = loss_by_iteration[0]
        mse_loss = get_mse_loss(loss_by_iteration[1], cache[solution_label][:,0].flatten())
    else:
        f_sol = y_pred
        print(params)
        optimal_alpha = params


    
    
    mse = np.mean((f_sol - cache[solution_label][:,0].flatten()))**2

    dict_to_save = {"f_sol": f_sol, 
                    "optimal_alpha": optimal_alpha, 
                    "mse": mse, 
                    "method": experiment["method"],
                    "loss_name": experiment["loss_name"],
                    "domain": experiment["x_domain"], 
                    "executor_type": executor_str,
                    "encoding_circuit": encoding_circuit_label,
                    **experiment["circuit_information"],
    }
    
    try:
        dict_to_save["mse_history"] = mse_loss
        dict_to_save["loss_history"] = ode_loss
    except:
        pass
    
    for key, value in experiment.items():
        if key == "circuit_information" or "loss" or "derivatives_of_loss" or "grad_loss" or "executor_type":
            pass
        else:
            dict_to_save[key] = value

    if experiment["method_information"] is not None:
        for key, value in experiment["method_information"].items():
            dict_to_save[key] = value
   

    print(dict_to_save)
    
    #result_dic_list.append(pd.DataFrame([dict_to_save]))

    df_ = pd.DataFrame([dict_to_save])

    #df = pd.concat(df_, axis=0, sort=False, ignore_index=True)
    #df.reset_index(drop=True, inplace=True)

    df_.to_feather(experiment_path + ".feather")
    print("Saved at:", experiment_path + ".feather")
