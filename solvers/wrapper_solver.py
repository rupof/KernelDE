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


class CircuitInformation:
    def __init__(self, experiment):
        self.x_span = experiment["x_domain"] * experiment["quantum_bandwidth"]
        self.method = experiment["method"]
        # self.eta = experiment["method_information"]["eta"]
        # self.quantum_bandwidth = experiment["quantum_bandwidth"]
        self.executor_str = experiment["executor_type"]
        self.encoding_circuit_label = (
            "NoCircuit"
            if experiment["circuit_information"]["encoding_circuit"] == "NoCircuit"
            else experiment["circuit_information"]["encoding_circuit"].__name__
        )
        #self.experiment_path = experiment["path"]
        self.executor_object = executor_type_dictionary[self.executor_str]
        self.num_qubits = experiment["circuit_information"]["num_qubits"]
        self.circuit_information = experiment["circuit_information"]
    def get_info(self):
        return {"x_span": self.x_span, "method": self.method, "executor_str": self.executor_str, "encoding_circuit_label": self.encoding_circuit_label, "num_qubits": self.num_qubits}
    
    def __eq__(self, other):
        if isinstance(other, CircuitInformation):
            return np.array_equal(self.x_span, other.x_span) and self.method == other.method and self.executor_str == other.executor_str and self.encoding_circuit_label == other.encoding_circuit_label and self.num_qubits == other.num_qubits
        return False
    def __hash__(self):
        return hash((tuple(self.x_span), self.method, self.executor_str, self.encoding_circuit_label, self.num_qubits))


#np.mean(((y[:]-1)**2)[:,0], axis=1)
def get_f_loss(f_by_iteration, y_exact, axis = 1):
    return np.mean((f_by_iteration-y_exact)**2, axis = axis)
   
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

    f_solution_label = f"f_{experiment['loss_name']}"
    dfdx_solution_label = f"dfdx_{experiment['loss_name']}"


    if f_solution_label in cache:
        f_numerical_solution = cache[f_solution_label] #f_numerical_solution: shape (x_span.shape[0],)
        dfdx_numerical_solution = cache[dfdx_solution_label]
    else:
        f_numerical_solution = odeint(experiment["derivatives_of_loss"], f_initial, x_span[:]).flatten()
        cache[f_solution_label] = f_numerical_solution
        dfdx_numerical_solution = np.gradient(f_numerical_solution, x_span)
        cache[dfdx_solution_label] = dfdx_numerical_solution

    try:
        eta = experiment["method_information"]["eta"]
    except:
        eta = 1

    #print experiment details


    if experiment["method"] == "PQK":
        #get experiment["method_information"]["eta"] if not 1
        
        OSolver = PQK_solver(experiment["circuit_information"],
                                executor_object, 
                                envelope={"function": matrix_rbf, 
                                            "derivative_function": matrix_rbf_dx_slow, 
                                            "second_derivative_function": matrix_rbf_dxdx_slow,
                                               "mixed_derivative_function": matrix_rbf_dxdy_slow,
                                            "sigma": experiment["sigma"]}, 
                                regularization_parameter=eta,
                                CircuitInformation = CircuitInformation(experiment))
        dict_to_save = {"sigma": experiment["sigma"]}
        experiment["circuit_information"].pop("encoding_circuit")

    elif experiment["method"] == "FQK":
        OSolver = FQK_solver(experiment["circuit_information"],
                                executor_object, regularization_parameter=eta, CircuitInformation = CircuitInformation(experiment))
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

        loss_ODE = ODELoss(loss, grad_loss, initial_vec = f_initial, eta=eta, boundary_handling = boundary_handling, true_solution = f_numerical_solution)
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
            variance = variance_for_qnn_regularization
        )    

        y_ODE = np.zeros((x_span.shape[0]))
        clf._fit(x_span, y_ODE,  weights=None)
        y_pred = clf.predict(x_span)
        params = clf._param
    
    if experiment["method"].startswith("QNN") == False:    
        solution, loss_by_iteration = OSolver.solver(x_span, f_initial, loss)
        f_sol = solution[0]
        optimal_alpha = solution[1]
        L_loss_history = loss_by_iteration[0]  #shape (iteration, x_span.shape[0] )
        f_loss_history = get_f_loss(np.array(loss_by_iteration[1]), f_numerical_solution) 
        iv_loss_history = eta*(np.array(loss_by_iteration[1])[:,0] - f_initial)**2   
        dfdx_loss_history = get_f_loss(np.array(loss_by_iteration[2]), dfdx_numerical_solution)
        #print initial and final loss
        # print(f"Initial L Loss: {L_loss_history[0]}", f"Final L Loss: {L_loss_history[-1]}")
        # print(f"Initial f Loss: {f_loss_history[0]}", f"Final f Loss: {f_loss_history[-1]}")
        # print(f"Initial dfdx Loss: {dfdx_loss_history[0]}", f"Final dfdx Loss: {dfdx_loss_history[-1]}")
        # print(f"Initial iv Loss: {iv_loss_history[0]}", f"Final iv Loss: {iv_loss_history[-1]}")
        print("----")

    else:
        f_sol = y_pred
        print(params)
        optimal_alpha = params
    
    mse = np.mean((f_sol - f_numerical_solution)**2)

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
        dict_to_save["f_loss_history"] = f_loss_history #old name:mse_history
        dict_to_save["dfdx_loss_history"] = dfdx_loss_history 
        dict_to_save["iv_loss_history"] = iv_loss_history
        dict_to_save["L_loss_history"] = L_loss_history #L_loss_history
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
   

    #print(dict_to_save)
    
    #result_dic_list.append(pd.DataFrame([dict_to_save]))

    df_ = pd.DataFrame([dict_to_save])

    #df = pd.concat(df_, axis=0, sort=False, ignore_index=True)
    #df.reset_index(drop=True, inplace=True)

    df_.to_feather(experiment_path + ".feather")
    #print("Saved at:", experiment_path + ".feather")
