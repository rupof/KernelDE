import numpy as np
import sys
from itertools import product
from circuits.circuits import * 
from DE_Library.diferential_equation_functionals import *
from squlearn.qnn import get_lr_decay





def get_experiment_combination_list(experimental_parameters):
    """
    Experimental parameters is a list of lists. Each list contains the possible values for a parameter.

    for example: 
    
    experimental_parameters = [dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list, ...]
    """
    experiment_list = []
    sorted_parameters = experimental_parameters
    for params in product(*sorted_parameters):
        function_pair, encoding_circuit, num_qubits, num_layers, sigma, method, executor_type, quantum_bandwidth = params
        
        loss_name = function_pair[0]
        f_initial = function_pair[1]
        x_domain = function_pair[2]

        if len(method) == 2:
            method_name = method[0] 
            method_info = method[1]

        else:
            method_name = method
            method_info = None
            
        
        experiment = {"circuit_information": {
            "encoding_circuit": circuits_dictionary_qiskit[encoding_circuit],
            "num_qubits": num_qubits,
            "num_layers": num_layers
            },
            "sigma": sigma,
            "loss": mapping_of_loss_functions[loss_name],
            "derivatives_of_loss": mapping_of_derivatives_of_loss_functions[loss_name],
            "grad_loss": mapping_of_grad_of_loss_functions[loss_name],
            "method": method_name,
            "loss_name": loss_name,
            "executor_type": executor_type,
            "f_initial": f_initial,
            "quantum_bandwidth": quantum_bandwidth,
            "x_domain": x_domain,
            "method_information": method_info,
        }
        experiment_list.append(experiment)
    return experiment_list



function_list = [("paper", [1], np.linspace(0.0001, 1.5*3.14, 50))]
num_qubits_list = [2, 3]
num_layers_list = [1, 2]
gamma_classical_bandwidth_list = np.linspace(0.1, 5, 2)
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
encoding_circuit_list = ["HardwareEfficientEmbeddingCircuit", "Separable_rx"]
executor_type_list = ["statevector_simulator"]    
quantum_bandwith = [1]

method_list = ["PQK"]
experiment_first_combination = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, method_list, executor_type_list, quantum_bandwith])


function_list = [("paper", [1], np.linspace(0.0001, 1.5*3.14, 50))]
num_qubits_list = [2, 3, 4, 5, 6]
num_layers_list = [1, 2, 3]
gamma_classical_bandwidth_list = np.linspace(0.1, 5, 25)
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
encoding_circuit_list = ["HardwareEfficientEmbeddingCircuit", "Separable_rx"]
executor_type_list = ["statevector_simulator"]    

method_list = ["PQK"]
experiment_better_combination = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, method_list, executor_type_list, quantum_bandwith])
experiment_2_combination = get_experiment_combination_list([function_list, encoding_circuit_list, [2, 3, 4, 5, 6, 7, 8, 9, 10, 12], num_layers_list, np.linspace(0.1, 3, 40), method_list, executor_type_list, quantum_bandwith])
experiment_2_FQK_combination = get_experiment_combination_list([function_list, encoding_circuit_list, [2, 3, 4, 5, 6, 7, 8, 9, 10, 12], num_layers_list,[0], ["FQK"], executor_type_list, quantum_bandwith])


#########33
function_list = [("paper", [1], np.linspace(0.0001, 1.5*3.14, 50)), 
                 ("log_ode", [np.log(0.0001)], np.linspace(0.0001, 1.5*3.14, 50)), 
                 ("polynomial_with_exp", [3], np.linspace(0.0001, 1.5*3.14, 50))]
num_qubits_list = [2, 4, 6]
num_layers_list = [1, 5, 10]
experiment_FQK_combination_starting = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, [0], ["FQK"], executor_type_list, quantum_bandwith])


#Classical RBF
function_list = [("paper", [1], np.linspace(0.0001, 1.5*3.14, 50)), 
                 ("log_ode", [np.log(0.0001)], np.linspace(0.0001, 1.5*3.14, 50)), 
                 ("polynomial_with_exp", [3], np.linspace(0.0001, 1.5*3.14, 50))]
num_qubits_list = [2]
num_layers_list = [1]
gamma_classical_bandwidth_list = np.linspace(0.1, 5, 200)
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
experiment_RBF_combination_starting = get_experiment_combination_list([function_list, ["NoCircuit"], [0], [0], sigma_classical_bandwidth_list, ["classical_RBF"], executor_type_list, quantum_bandwith])

#PQK 
function_list = [("paper", [1], np.linspace(0.0001, 1.5*3.14, 50)), 
                 ("log_ode", [np.log(0.0001)], np.linspace(0.0001, 1.5*3.14, 50)), 
                 ("polynomial_with_exp", [3], np.linspace(0.0001, 1.5*3.14, 50))]
num_qubits_list = [2, 4, 6]
num_layers_list = [1, 5, 10]
gamma_classical_bandwidth_list = np.linspace(0.1, 5, 20)
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
experiment_PQK_combination_starting = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, ["PQK"], executor_type_list, quantum_bandwith])


function_list_ho = [("harmonic_oscillator", [np.cos(-0.9), -np.sin(0.9)], np.linspace(-0.9, 0.9, 22))]
encoding_circuit_list = ["Separable_rx", "HEEAndChebyshevTower"]
num_qubits_list = [2, 4, 6]
num_layers_list = [1]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
experiment_PQK_combination_ho = get_experiment_combination_list([function_list_ho, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("PQK")], executor_type_list, quantum_bandwith])

###############


encoding_circuit_list = ["Separable_rx"]
function_list_ho = [("harmonic_oscillator", [1, -1], np.linspace(0, 1*3.14, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [2]
num_layers_list = [1]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
experiment_QNN_combination_ho = get_experiment_combination_list([function_list_ho, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("PQK"), ("FQK")], executor_type_list, quantum_bandwith])
#8

encoding_circuit_list = ["YZ_CX_EncodingCircuit"]
function_list = [("paper", [1], np.linspace(0.0001, 2, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [8]
num_layers_list = [3]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
method_information = {"eta": 1, 
                        "boundary_handling": "pinned", 
                        "optimizer": "Adam", 
                        "lr": 0.05, 
                        "maxiter": 400, 
                        "tol": 1e-4}
method_list = [("QNN", method_information)]
experiment_QNN_combination_paper = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, method_list, executor_type_list, quantum_bandwith])

encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("paper", [1], np.linspace(0, 0.9, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [6]
num_layers_list = [5]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
method_information_pinned = {"eta": 1, 
                        "boundary_handling": "pinned", 
                        "optimizer": "Adam", 
                        "lr": 0.05, 
                        "maxiter": 100, 
                        "tol": 1e-4}
method_information_floating = {"eta": 1, 
                        "boundary_handling": "floating", 
                        "optimizer": "Adam", 
                        "lr": 0.05, 
                        "maxiter": 100, 
                        "tol": 1e-4}

method_list = [("QNN", method_information_floating), ("QNN", method_information_pinned)]
experiment_QNN_combination_paper_Chebyshev = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [method_information_pinned, method_information_floating], executor_type_list, quantum_bandwith])
#10



encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("paper_decay_QNN", [1], np.linspace(0, 0.9, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [6]
num_layers_list = [5]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
method_list = [("QNN", method_information_floating)]
experiment_QNN_decay_combination_paper_Chebyshev = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, method_list, executor_type_list, quantum_bandwith])
#11

encoding_circuit_list = ["ChebyshevTowerAndHEE_repeat"]
function_list = [("paper_decay_QNN", [1], np.linspace(0, 0.9, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [6]
num_layers_list = [5]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
experiment_QNN_decay_combination_paper_floating_Chebyshev = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("QNN", method_information_pinned)], executor_type_list, quantum_bandwith])
#12

encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("paper_decay_QNN", [1], np.linspace(0, 0.9, 22))]
executor_type_list = ["pennylane"]    
num_qubits_list = [6]
num_layers_list = [5]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
method_information_pinned = {"eta": 1, 
                        "boundary_handling": "pinned", 
                        "optimizer": "Adam", 
                        "lr": 0.05, 
                        "maxiter": 1000, 
                        "tol": 1e-4}
method_information_floating = {"eta": 1, 
                        "boundary_handling": "floating", 
                        "optimizer": "Adam", 
                        "lr": 0.05, 
                        "maxiter": 1000, 
                        "tol": 1e-4}

method_list = [("QNN", method_information_floating), ("QNN", method_information_pinned)]
experiment_QNN_paper_decay_test = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, method_list, executor_type_list, quantum_bandwith])
#13

encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("simple_test_QNN", [1], np.linspace(0, 0.9, 4))]
executor_type_list = ["pennylane"]    
num_qubits_list = [3] #6
num_layers_list = [1] #5
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
method_information_pinned = {"eta": 1, 
                        "boundary_handling": "pinned", 
                        "optimizer": "Adam", 
                        "lr": [get_lr_decay(0.05, 0.001, 10)(i) for i in range(0, 11)], 
                        "maxiter": 10, 
                        "tol": 1e-4,}
method_information_pinned = {"eta": 1, 
                        "boundary_handling": "pinned", 
                        "optimizer": "SGLBO", 
                        "lr": 0.05, 
                        "maxiter": 10, 
                        "tol": 1e-4,}

method_list = [("QNN", method_information_floating), ("QNN", method_information_pinned)]
experiment_QNN_test_with_RX = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("QNN", method_information_pinned)], executor_type_list, quantum_bandwith])
#14


encoding_circuit_list = ["ChebyshevTowerAndHEE_repeat"]
function_list = [("log_ode", [np.log(0.001)], np.linspace(0.001, 0.9, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [6]
num_layers_list = [5]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
QNN_log_test = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("QNN", method_information_pinned)], executor_type_list, quantum_bandwith])

encoding_circuit_list = ["ChebyshevTowerAndHEE_repeat"]
function_list = [("log_ode", [np.log(0.01)], np.linspace(0.01, 0.9, 20)), ("simple_test_QNN", [1], np.linspace(0, 0.9, 20)), ("harmonic_oscillator", [1, 0], np.linspace(0, 0.9, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [2,6,8]
num_layers_list = [5, 10]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
big_experiment = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("QNN", method_information_pinned)], executor_type_list, quantum_bandwith])



encoding_circuit_list = ["SimpleAnalyticalCircuit"]
function_list = [ ("harmonic_oscillator", [np.cos(0), -np.sin(0)], np.linspace(-2*3.14, 2*3.14, 40)) ]
executor_type_list = ["pennylane"]    
num_qubits_list = [1]
num_layers_list = [3]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
experiment_QNN_combination_ho_ppt = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("QNN", method_information_pinned)], executor_type_list, quantum_bandwith])

encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("log_ode", [np.log(0.01)], np.linspace(0.01, 0.9, 20))]
executor_type_list = ["pennylane_shots_variance", "pennylane_shots", "pennylane"]    
num_qubits_list = [5]
num_layers_list = [3]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
log_experiment_shots_and_variance = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("QNN", method_information_pinned)], executor_type_list, quantum_bandwith])
#18

encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("log_ode", [np.log(0.01)], np.linspace(0.01, 0.9, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [4, 6]
num_layers_list = [3]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
log_experiment_chebyshev = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("QNN", method_information_pinned), ("QNN", method_information_floating) ], executor_type_list, quantum_bandwith])
#19

encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("log_ode", [np.log(0.01)], np.linspace(0.01, 0.9, 12))]
executor_type_list = ["qasm_simulator_variance", "qasm_simulator", "pennylane"]    
num_qubits_list = [6]
num_layers_list = [3]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
log_experiment_shots_and_variance_qiskit = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("QNN", method_information_pinned)], executor_type_list, quantum_bandwith])


encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("log_ode", [np.log(0.01)], np.linspace(0.01, 0.9, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [8]
num_layers_list = [4]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
method_information_pinned = {"eta": 1, 
                        "boundary_handling": "pinned", 
                        "optimizer": "Adam", 
                        "lr": 0.05, 
                        "maxiter": 1500, 
                        "tol": 1e-4}
log_experiment_chebyshev_iteration = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("QNN", method_information_pinned)], executor_type_list, quantum_bandwith])
#21


encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("damped_harmonic_oscillator", [1, -1], np.linspace(0, 0.9, 40))]
executor_type_list = ["pennylane"]    
num_qubits_list = [6]
num_layers_list = [5]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
damped_HO_experiment_chebyshev = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("QNN", method_information_pinned)], executor_type_list, quantum_bandwith])
#22


encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("paper", [1], np.linspace(0, 0.9, 20))]
executor_type_list = ["pennylane_shots_variance",   "pennylane_shots", "pennylane"]
num_qubits_list = [6]
num_layers_list = [5]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
exp_23 = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("QNN", method_information_pinned)], executor_type_list, quantum_bandwith])


function_list = [ ("log_ode", [np.log(0.01)], np.linspace(0.01, 0.9, 20)), ]
kernel_experiment_log1_for_ppt = get_experiment_combination_list([function_list, 
                                                                   ["Separable_rx"], 
                                                                  [8], [2], [1.5], 
                                                                  ["PQK"], ["pennylane"], quantum_bandwith])
function_list = [ ("log_ode", [np.log(0.01)], np.linspace(0.01, 0.9, 20)), ]
kernel_experiment_log2_for_ppt = get_experiment_combination_list([function_list, 
                                                                   ["Separable_rx"], #HardwareEfficientEmbeddingCircuit
                                                                  [7], [2], [1.5], 
                                                                  ["FQK"], ["pennylane"], quantum_bandwith])

function_list = [ ("log_ode", [np.log(0.01)], np.linspace(0.01, 0.9, 20)), ]
kernel_experiment_log3_for_ppt = get_experiment_combination_list([function_list, 
                                                                   ["NoCircuit"], 
                                                                  [7], [2], [0.4], 
                                                                  ["classical_RBF"], ["pennylane"], quantum_bandwith])


function_list = [ ("harmonic_oscillator", [np.cos(0), -np.sin(0)], np.linspace(-2*3.14, 2*3.14, 40)), ]
kernel_experiment_sho_for_ppt = get_experiment_combination_list([function_list, 
                                                                   ["HardwareEfficientEmbeddingCircuit"], 
                                                                  [3], [1], [1.5], 
                                                                  ["FQK"], ["pennylane"], quantum_bandwith])

kernel_experiment_sho1_for_ppt = get_experiment_combination_list([function_list, 
                                                                   ["NoCircuit"], 
                                                                  [7], [2], [2.5], 
                                                                  ["classical_RBF"], ["pennylane"], quantum_bandwith])


encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("paper", [1], np.linspace(0, 0.9, 20)), ("log_ode", [np.log(0.01)], np.linspace(0.01, 0.9, 20))]
executor_type_list = ["pennylane"]
num_qubits_list = [2, 3, 4, 5, 6]
num_layers_list = [5]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
method_information_pinned = {"eta": 1, 
                        "boundary_handling": "pinned", 
                        "optimizer": "Adam", 
                        "lr": 0.05, 
                        "maxiter": 120, 
                        "tol": 1e-4}
method_information_floating = {"eta": 1, 
                        "boundary_handling": "floating", 
                        "optimizer": "Adam", 
                        "lr": 0.05, 
                        "maxiter": 120, 
                        "tol": 1e-4}
exp_23 = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("QNN", method_information_pinned), ("QNN", method_information_floating)], executor_type_list, quantum_bandwith])

method_information_pinned = {"eta": 1, 
                        "boundary_handling": "pinned", 
                        "optimizer": "Adam", 
                        "lr": 0.03, 
                        "maxiter": 300, 
                        "tol": 1e-4}
method_information_floating = {"eta": 1, 
                        "boundary_handling": "floating", 
                        "optimizer": "Adam", 
                        "lr": 0.03, 
                        "maxiter": 300, 
                        "tol": 1e-4}

method_information_pinned_decay = {"eta": 1, 
                        "boundary_handling": "pinned", 
                        "optimizer": "Adam", 
                        "lr": [get_lr_decay(0.1, 0.01, 300)(i) for i in range(0, 301)], 
                        "maxiter": 300, 
                        "tol": 1e-4}
method_information_floating_decay = {"eta": 1, 
                        "boundary_handling": "floating", 
                        "optimizer": "Adam", 
                        "lr": [get_lr_decay(0.1, 0.01, 300)(i) for i in range(0, 301)], 
                        "maxiter": 300, 
                        "tol": 1e-4}

encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("paper", [1], np.linspace(0, 0.9, 30)), ("log_ode", [np.log(0.01)], np.linspace(0.01, 0.9, 30))]
executor_type_list = ["pennylane"]
num_qubits_list = [2, 3, 4, 5, 6, 7, 8]
num_layers_list = [1, 3, 6, 9]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
exp_long_qnn = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("QNN", method_information_pinned), 
                                                                                                                                                         ("QNN", method_information_floating),
                                                                                                                                                         ("QNN", method_information_floating_decay),
                                                                                                                                                         ("QNN", method_information_pinned_decay)], executor_type_list, quantum_bandwith])


encoding_circuit_list = ["HEEAndChebyshevTower"]
function_list = [("paper", [1], np.linspace(0, 0.9, 30)), ("log_ode", [np.log(0.01)], np.linspace(0.01, 0.9, 30))]
executor_type_list = ["pennylane"]
num_qubits_list = [2, 3, 4, 5, 6, 7, 8]
num_layers_list = [1, 3, 6, 9]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
exp_long_kernel = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("FQK"), ("PQK")
                                                                                                                                                            ], executor_type_list, quantum_bandwith])


encoding_circuit_list = ["HEEAndChebyshevTower"]
function_list = [("paper", [1], np.linspace(0, 0.9, 30)), ("log_ode", [np.log(0.01)], np.linspace(0.01, 0.9, 30))]
executor_type_list = ["pennylane"]
num_qubits_list = [2, 3, 4, 5, 6, 7, 8]
num_layers_list = [1, 3, 6, 9]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
exp_long_only_PQK_kernel = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("PQK")
                                                                                                                                                            ], executor_type_list, quantum_bandwith])

encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("paper", [1], np.linspace(0, 0.9, 20))]
executor_type_list = ["pennylane_shots_variance",   "pennylane_shots"]
num_qubits_list = [6]
num_layers_list = [5]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
method_information_floating_1 = {"eta": 1, 
                        "boundary_handling": "floating", 
                        "optimizer": "Adam", 
                        "lr": 0.05, 
                        "maxiter": 250, 
                        "tol": 1e-4,  
                        "num_shots": 5000}
method_information_floating_2 = {"eta": 1, 
                        "boundary_handling": "floating", 
                        "optimizer": "Adam", 
                        "lr": 0.05, 
                        "maxiter": 250, 
                        "tol": 1e-4,  
                        "num_shots": 8000}
method_information_floating_no_shots = {"eta": 1, 
                        "boundary_handling": "floating", 
                        "optimizer": "Adam", 
                        "lr": 0.05, 
                        "maxiter": 250, 
                        "tol": 1e-4,  }

experiment_shots_prob_to_show = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("QNN", method_information_floating_1), ("QNN", method_information_floating_2)], executor_type_list, quantum_bandwith])
experiment_shots_prob_to_show_no_shot = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("QNN", method_information_floating_no_shots)], ["pennylane"], quantum_bandwith])
#concatenate both arrays
experiment_shots_prob = experiment_shots_prob_to_show #+ experiment_shots_prob_to_show_no_shot
#31


encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("paper_decay_QNN", [1], np.linspace(0, 0.9, 22))]
executor_type_list = ["pennylane"]    
num_qubits_list = [6]
num_layers_list = [5]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2

method_information_floating_list = {"eta": 1, 
                        "boundary_handling": "floating", 
                        "optimizer": "Adam", 
                        "lr": [get_lr_decay(0.05, 0.001, 250)(i) for i in range(0, 251)],
                        "maxiter": 250, 
                        "tol": 1e-5}

method_information_floating_small_adam = {"eta": 1, 
                        "boundary_handling": "floating", 
                        "optimizer": "Adam", 
                        "lr": 0.05, 
                        "maxiter": 250, 
                        "tol": 1e-5}

method_information_floating_small_sglbo = {"eta": 1, 
                        "boundary_handling": "floating", 
                        "optimizer": "SGLBO", 
                        "lr": 0.05, 
                        "maxiter": 250, 
                        "tol": 1e-5}

method_information_floating_ = {"eta": 1, 
                        "boundary_handling": "floating", 
                        "optimizer": "SGLBO", 
                        "lr": [get_lr_decay(0.05, 0.001, 250)(i) for i in range(0, 251)],
                        "maxiter": 250, 
                        "tol": 1e-5}


#method_list = [("QNN", method_information_floating_list), ("QNN", method_information_floating_small_adam), ("QNN", method_information_floating_small_sglbo), ("QNN", method_information_floating_)]
method_list = [("QNN", method_information_floating_list), ("QNN", method_information_floating_small_adam),]

experiment_QNN_paper_decay_test_benchmark = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, method_list, executor_type_list, quantum_bandwith])
#32

encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("paper", [1], np.linspace(0, 0.9, 4))]
executor_type_list = ["pennylane_shots_variance",   "pennylane_shots"]
num_qubits_list = [2]
num_layers_list = [1]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
method_information_floating_1 = {"eta": 1, 
                        "boundary_handling": "floating", 
                        "optimizer": "Adam", 
                        "lr": 0.05, 
                        "maxiter": 5, 
                        "tol": 1e-4,  
                        "num_shots": 5000}


experiment_shots_fast_test = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, [("QNN", method_information_floating_1)], executor_type_list, quantum_bandwith])
#concatenate both arrays
#31

experiment_list_total = [experiment_first_combination, #0
                        experiment_better_combination, #1
                        experiment_2_combination, #2
                        experiment_2_FQK_combination, #3
                        experiment_FQK_combination_starting, #4
                        experiment_RBF_combination_starting, # 5
                        experiment_PQK_combination_starting, #6
                        experiment_PQK_combination_ho, #7
                        experiment_QNN_combination_ho, #8
                        experiment_QNN_combination_paper, #9
                        experiment_QNN_combination_paper_Chebyshev, #10
                        experiment_QNN_decay_combination_paper_Chebyshev, #11
                        experiment_QNN_decay_combination_paper_floating_Chebyshev, #12
                        experiment_QNN_paper_decay_test,  #13
                        experiment_QNN_test_with_RX, #14
                        QNN_log_test, #15
                        big_experiment, #16
                        experiment_QNN_combination_ho_ppt, #17
                        log_experiment_shots_and_variance, #18
                        log_experiment_chebyshev, #19
                        log_experiment_shots_and_variance_qiskit, #20
                        log_experiment_chebyshev_iteration, #21
                        damped_HO_experiment_chebyshev, #22
                        exp_23, #23
                        kernel_experiment_log1_for_ppt, #24
                        kernel_experiment_log2_for_ppt, #25
                        kernel_experiment_log3_for_ppt, #26
                        kernel_experiment_sho_for_ppt, #27
                        kernel_experiment_sho1_for_ppt, #28
                        exp_long_qnn, #29
                        exp_long_kernel, #30
                        experiment_shots_prob, #31
                        experiment_QNN_paper_decay_test_benchmark, #32
                        exp_long_only_PQK_kernel, #33
                        experiment_shots_fast_test, #34
                        ] #26
                        

