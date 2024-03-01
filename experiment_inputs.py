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
    for params in product(*sorted_parameters):
        function, encoding_circuit, num_qubits, num_layers, sigma, method, executor_type = params

        experiment = {"circuit_information": {
            "encoding_circuit": circuits_dictionary_qiskit[encoding_circuit],
            "num_qubits": num_qubits,
            "num_layers": num_layers
            },
            "sigma": sigma,
            "g": function,
            "method": method,
            "executor_type": executor_type_dictionary[executor_type],
        }
        experiment_list.append(experiment)
    return experiment_list


def g(f, x):
        lamb = 20
        k = 0.1
        return -lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) - lamb * k * f

function_list = [g]
num_qubits_list = [2, 3]
num_layers_list = [1, 2]
gamma_classical_bandwidth_list = np.linspace(0.1, 5, 2)
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
encoding_circuit_list = ["HardwareEfficientEmbeddingCircuit", "Separable_rx"]
executor_type_list = ["statevector_simulator"]    

method_list = ["PQK"]
experiment_first_combination = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, method_list, executor_type_list])


function_list = [g]
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

experiment_list_total = [experiment_first_combination, #0
                        experiment_better_combination, #1
                        experiment_2_combination, #2
                        experiment_2_FQK_combination, #3
                        ] #1