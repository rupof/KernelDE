import numpy as np
import sys
from itertools import product
from circuits.circuits import * 



executor_type_dictionary = {
    "statevector_simulator": Executor("statevector_simulator"),
}

def get_experiment_combination_list(experimental_parameters):
    """
    Experimental parameters is a list of lists. Each list contains the possible values for a parameter.

    for example: 
    
    experimental_parameters = [dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list, ...]
    """
    experiment_list = []
    sorted_parameters = experimental_parameters
    print(sorted_parameters)
    for params in product(*sorted_parameters):
        function_pair, encoding_circuit, num_qubits, num_layers, sigma, method, executor_type = params
        function = function_pair[0]
        f_initial = function_pair[1]

        experiment = {"circuit_information": {
            "encoding_circuit": circuits_dictionary_qiskit[encoding_circuit],
            "num_qubits": num_qubits,
            "num_layers": num_layers
            },
            "sigma": sigma,
            "g": mapping_of_g_functions[function],
            "method": method,
            "g_name": function,
            "executor_type": executor_type_dictionary[executor_type],
            "f_initial": f_initial
        }
        experiment_list.append(experiment)
    return experiment_list


def g_paper(f, x):
        """
        df/dx = -lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) - lamb * k * f

        solution: f(x) = np.exp(-lamb * x * k) * np.cos(lamb * x), f(0) = 1
        """
        lamb = 20
        k = 0.1
        return -lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) - lamb * k * f

def g_exp(f, x):
    """
    df/dx = lamb * np.exp(f * k) 
    f(0.001) = np.log(0.001)

    solution: f(x) = np.log(x)
    """
    lamb = 1
    k = 1
    return np.exp(-f*k)*lamb

def g_exp_2(f, x):
    """
    df/dx = 2*f+4*cos(x)-8*sin(x), f(0) = 3

    solution: f(x) = 3*exp(2*x) + 4*sin(x)
    """
    return 2*f+4*np.cos(x)-8*np.sin(x)

mapping_of_g_functions = {
    "g_paper": g_paper,
    "g_exp": g_exp,
    "g_exp_2": g_exp_2
}



function_list = [("g_paper", 1)]
num_qubits_list = [2, 3]
num_layers_list = [1, 2]
gamma_classical_bandwidth_list = np.linspace(0.1, 5, 2)
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
encoding_circuit_list = ["HardwareEfficientEmbeddingCircuit", "Separable_rx"]
executor_type_list = ["statevector_simulator"]    

method_list = ["PQK"]
experiment_first_combination = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, method_list, executor_type_list])


function_list = [("g_paper", 1)]
num_qubits_list = [2, 3, 4, 5, 6]
num_layers_list = [1, 2, 3]
gamma_classical_bandwidth_list = np.linspace(0.1, 5, 25)
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
encoding_circuit_list = ["HardwareEfficientEmbeddingCircuit", "Separable_rx"]
executor_type_list = ["statevector_simulator"]    

method_list = ["PQK"]
experiment_better_combination = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, method_list, executor_type_list])
experiment_2_combination = get_experiment_combination_list([function_list, encoding_circuit_list, [2, 3, 4, 5, 6, 7, 8, 9, 10, 12], num_layers_list, np.linspace(0.1, 3, 40), method_list, executor_type_list])
experiment_2_FQK_combination = get_experiment_combination_list([function_list, encoding_circuit_list, [2, 3, 4, 5, 6, 7, 8, 9, 10, 12], num_layers_list,[0], ["FQK"], executor_type_list])


#########33
function_list = [("g_paper", 1), ("g_exp", np.log(0.0001)), ("g_exp_2", 3)]
num_qubits_list = [2, 4, 6]
num_layers_list = [1, 5, 10]
experiment_FQK_combination_starting = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, [0], ["FQK"], executor_type_list])


#Classical RBF
function_list = [("g_paper", 1), ("g_exp", np.log(0.0001)), ("g_exp_2", 3)]
num_qubits_list = [2]
num_layers_list = [1]
gamma_classical_bandwidth_list = np.linspace(0.1, 5, 200)
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
experiment_RBF_combination_starting = get_experiment_combination_list([function_list, ["NoCircuit"], [0], [0], sigma_classical_bandwidth_list, ["classical_RBF"], executor_type_list])

#PQK 
function_list = [("g_paper", 1), ("g_exp", np.log(0.0001)), ("g_exp_2", 3)]
num_qubits_list = [2, 4, 6]
num_layers_list = [1, 5, 10]
gamma_classical_bandwidth_list = np.linspace(0.1, 5, 20)
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
experiment_PQK_combination_starting = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, ["PQK"], executor_type_list])


###############


experiment_list_total = [experiment_first_combination, #0
                        experiment_better_combination, #1
                        experiment_2_combination, #2
                        experiment_2_FQK_combination, #3
                        experiment_FQK_combination_starting, #4
                        experiment_RBF_combination_starting, # 5
                        experiment_PQK_combination_starting #6

                        ] 

