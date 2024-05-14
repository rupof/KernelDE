import numpy as np
import sys
from itertools import product
from circuits.circuits import * 
from DE_Library.diferential_equation_functionals import *



executor_type_dictionary = {
    "statevector_simulator": Executor("statevector_simulator"),
    "pennylane": Executor("pennylane"), 
    "qasm_simulator_variance": Executor("qasm_simulator", shots=5000, seed=1),
    "pennylane_shots_variance": Executor("default.qubit", shots=7000, seed = 1),
    "qasm_simulator": Executor("qasm_simulator", shots=5000, seed=1),
    "pennylane_shots": Executor("default.qubit", shots=7000, seed = 1),
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
        function_pair, encoding_circuit, num_qubits, num_layers, sigma, method, executor_type, quantum_bandwidth = params
        loss_name = function_pair[0]
        f_initial = function_pair[1]
        x_domain = function_pair[2]
        
        experiment = {"circuit_information": {
            "encoding_circuit": circuits_dictionary_qiskit[encoding_circuit],
            "num_qubits": num_qubits,
            "num_layers": num_layers
            },
            "sigma": sigma,
            "loss": mapping_of_loss_functions[loss_name],
            "derivatives_of_loss": mapping_of_derivatives_of_loss_functions[loss_name],
            "grad_loss": mapping_of_grad_of_loss_functions[loss_name],
            "method": method,
            "loss_name": loss_name,
            "executor_type": executor_type_dictionary[executor_type],
            "f_initial": f_initial,
            "quantum_bandwidth": quantum_bandwidth,
            "x_domain": x_domain
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


function_list_ho = [("harmonic_oscillator", [0, 1], np.linspace(0.0001, 1.5*3.14, 50))]
encoding_circuit_list = ["Separable_rx"]
num_qubits_list = [2, 4, 6]
num_layers_list = [1]
quantum_bandwith = [0.25, 0.5, 0.75, 1]
gamma_classical_bandwidth_list = np.linspace(0.1, 5, 10)
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
experiment_PQK_combination_ho = get_experiment_combination_list([function_list_ho, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, ["PQK"], executor_type_list, quantum_bandwith])

###############


encoding_circuit_list = ["SimpleAnalyticalCircuit"]
function_list_ho = [("harmonic_oscillator", [1, 0], np.linspace(0, 1*3.14, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [1]
num_layers_list = [1]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
experiment_QNN_combination_ho = get_experiment_combination_list([function_list_ho, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, ["QNN_pinned"], executor_type_list, quantum_bandwith])


encoding_circuit_list = ["YZ_CX_EncodingCircuit"]
function_list = [("paper", [1], np.linspace(0.0001, 2, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [8]
num_layers_list = [3]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
experiment_QNN_combination_paper = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, ["QNN_pinned"], executor_type_list, quantum_bandwith])

encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("paper", [1], np.linspace(0, 0.9, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [2]
num_layers_list = [1]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
experiment_QNN_combination_paper_Chebyshev = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, ["QNN_floating"], executor_type_list, quantum_bandwith])
#10

encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("paper_decay_QNN", [1], np.linspace(0, 0.9, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [6]
num_layers_list = [5]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
experiment_QNN_decay_combination_paper_Chebyshev = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, ["QNN_floating"], executor_type_list, quantum_bandwith])


encoding_circuit_list = ["ChebyshevTowerAndHEE_repeat"]
function_list = [("paper_decay_QNN", [1], np.linspace(0, 0.9, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [6]
num_layers_list = [5]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
experiment_QNN_decay_combination_paper_floating_Chebyshev = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, ["QNN_pinned"], executor_type_list, quantum_bandwith])
#12

encoding_circuit_list = ["ChebyshevTowerAndHEE_repeat"]
function_list = [("paper_decay_QNN", [1], np.linspace(0, 0.9, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [6]
num_layers_list = [5]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
experiment_QNN_test = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, ["QNN_floating"], executor_type_list, quantum_bandwith])
#13

encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("simple_test_QNN", [1], np.linspace(0, 0.9, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [6]
num_layers_list = [5]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
experiment_QNN_test_with_RX = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, ["QNN_pinned"], executor_type_list, quantum_bandwith])
#14


encoding_circuit_list = ["ChebyshevTowerAndHEE_repeat"]
function_list = [("log_ode", [np.log(0.001)], np.linspace(0.001, 0.9, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [6]
num_layers_list = [5]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
QNN_log_test = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, ["QNN_pinned"], executor_type_list, quantum_bandwith])

encoding_circuit_list = ["ChebyshevTowerAndHEE_repeat"]
function_list = [("log_ode", [np.log(0.01)], np.linspace(0.01, 0.9, 20)), ("simple_test_QNN", [1], np.linspace(0, 0.9, 20)), ("harmonic_oscillator", [1, 0], np.linspace(0, 0.9, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [2,6,8]
num_layers_list = [5, 10]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
big_experiment = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, ["QNN_pinned"], executor_type_list, quantum_bandwith])



encoding_circuit_list = ["SimpleAnalyticalCircuit"]
function_list_ho = [("harmonic_oscillator", [1, 0], np.linspace(0, 1*3.14, 20))]
executor_type_list = ["qasm_simulator"]    
num_qubits_list = [1]
num_layers_list = [1]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
experiment_QNN_combination_ho_shots = get_experiment_combination_list([function_list_ho, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, ["QNN_pinned"], executor_type_list, quantum_bandwith])

encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("log_ode", [np.log(0.01)], np.linspace(0.01, 0.9, 20))]
executor_type_list = ["pennylane_shots_variance", "pennylane_shots"]    
num_qubits_list = [4]
num_layers_list = [3]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
log_experiment_shots_and_variance = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, ["QNN_pinned"], executor_type_list, quantum_bandwith])


encoding_circuit_list = ["ChebyshevTowerAndHEE"]
function_list = [("log_ode", [np.log(0.01)], np.linspace(0.01, 0.9, 20))]
executor_type_list = ["pennylane"]    
num_qubits_list = [8]
num_layers_list = [6]
quantum_bandwith = [1]
gamma_classical_bandwidth_list = np.array([1])
sigma_classical_bandwidth_list = 0.5*(1/gamma_classical_bandwidth_list)**2
log_experiment_chebyshev = get_experiment_combination_list([function_list, encoding_circuit_list, num_qubits_list, num_layers_list, sigma_classical_bandwidth_list, ["QNN_floating"], executor_type_list, quantum_bandwith])


#QNN_floating
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
                        experiment_QNN_test,  #13
                        experiment_QNN_test_with_RX, #14
                        QNN_log_test, #15
                        big_experiment, #16
                        experiment_QNN_combination_ho_shots, #17
                        log_experiment_shots_and_variance, #18
                        log_experiment_chebyshev, #19
                        ] 

