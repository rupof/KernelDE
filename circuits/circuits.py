from squlearn.encoding_circuit import *
from squlearn.encoding_circuit import HEE_rzrxrz

import numpy as np
#import reduce

from squlearn.util import Executor
from qiskit.primitives import Estimator, Sampler
from squlearn.kernel.matrix import FidelityKernel, ProjectedQuantumKernel
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
#import zzfeaturemap




def IQPLikeCircuit_qiskit(num_qubits, num_layers, inverse = False):
    """
    IQPLikeCircuit(num_qubits, num_layers)
    Returns a circuit that is similar to the one used in IQP.
    """
    from qiskit.circuit.library import ZZFeatureMap
    from functools import reduce

    def self_product(x: np.ndarray) -> float:
        """
        Define a function map from R^n to R.

        Args:
            x: data

        Returns:
            float: the mapped value
        """
        #product of all elements in x
        return reduce(lambda a, b: a * b, x)
    
    if inverse:
        return ZZFeatureMap(num_qubits, reps = num_layers, data_map_func = self_product).inverse()
    else:
        return ZZFeatureMap(num_qubits, reps = num_layers, data_map_func = self_product)

def IQPLikeCircuit(num_qubits, num_layers, inverse = False):
    """
    IQPLikeCircuit(num_qubits, num_layers)
    Returns a circuit that is similar to the one used in IQP.
    """
    return QiskitEncodingCircuit(IQPLikeCircuit_qiskit(num_qubits, num_layers, inverse))

#def Separable_rx(num_qubits, num_layers):
#    """
#    #Separable_rx(num_qubits, num_layers)
#    #Returns a circuit that is similar to the one used in IQP.
#    """
#    fmap = LayeredEncodingCircuit(num_qubits=num_qubits, num_features=num_qubits)
#    for layer in range(num_layers):
#        fmap.Rx("x")
#    return fmap
#"""

def Separable_rx_qiskit(num_qubits, num_layers, inverse = False, only_one_variable = False):
    QC = QuantumCircuit(num_qubits)

    symbol = "y" if inverse else "x"
    features = ParameterVector(f"{symbol}", num_qubits)
    if only_one_variable:
        features = [features[0]]*num_qubits
    for layer in range(num_layers):
        # Apply single-qubit rotations
        for i in range(num_qubits):
            QC.rx(features[i], i)
    if inverse:
        return QC.inverse()
    else:
        return QC
    
def Separable_rx(num_qubits, num_layers, inverse = False):  
    return QiskitEncodingCircuit(Separable_rx_qiskit(num_qubits, num_layers, inverse))

def HardwareEfficientEmbeddingCircuit_qiskit(num_qubits, num_layers, rotation_gate = "rx", inverse = False, only_one_variable = False):
    QC = QuantumCircuit(num_qubits)

    def h_rz_gate(theta, qubit):
        QC.h(qubit)
        QC.rz(theta, qubit)

    gate_mapping = {
        'rx': QC.rx,
        'ry': QC.ry,
        'rz': QC.rz,
        'h_rz': h_rz_gate
    }

    def mapping(x, q_i):
            """Non-linear mapping for x: alpha*i*arccos(x)"""
            return q_i * x / 2

    rotation_func = gate_mapping.get(rotation_gate)
    if rotation_func is None:
        raise ValueError("Invalid rotation_gate value. Choose 'rx', 'ry', 'rz', or 'h_rz'.")
        
    symbol = "y" if inverse else "x"
    features = ParameterVector(f"{symbol}", num_qubits)
    if only_one_variable:
        features = [features[0]]*num_qubits
    for layer in range(num_layers):
        # Apply single-qubit rotations
        for i in range(num_qubits):
            rotation_func(mapping(features[i], i), i % num_qubits)
        # Apply entangling gates
        for i in range(num_qubits - 1):
            QC.cx(i, i+1)
    if inverse:
        return QC.inverse()
    else:
        return QC
    
def HardwareEfficientEmbeddingCircuit(num_qubits, num_layers, rotation_gate, inverse = False):
    return QiskitEncodingCircuit(HardwareEfficientEmbeddingCircuit_qiskit(num_qubits, num_layers, rotation_gate, inverse))
#fmap = HardwareEfficientEmbeddingCircuit(num_qubits=num_qubits, num_layers=num_layers)


def Hamiltonian_time_evolution_encoding_qiskit(num_qubits, trotter_time_T, evolve_time_t, inverse = False, only_one_variable = False):
    """
    Implements the Trotter formula, to time evolve a 1D Heisenberg chain Hamiltonian.

    The feature map is given by:

    |x\rangle = \left(\prod_{j=1}^{n} e^{-i \frac{t}{T}x_j H_j}\right) \otimes_{j=1}^{n+1} |\phi_j \rangle^{\otimes n+1}

    where H_j = X_j X_{j+1} + Y_j Y_{j+1} + Z_j Z_{j+1} is the jth term in the Hamiltonian, and |\phi_j \rangle is a random Haar state of 
    fixed seed.
    
    #This feature map represents a n_components-dimensional datapoint as a n_components+1-qubit quantum state

    Power of data paper uses 

    trotter_time_T = 20 # Trotter time is equivalent to layers of the circuit
    evolve_time_t = n_components/3

    """

    from qiskit.quantum_info import Pauli, SparsePauliOp, random_statevector
    from qiskit.circuit.library import PauliEvolutionGate

    n_components = num_qubits - 1

    symbol = "y" if inverse else "x"
    features = ParameterVector(f"{symbol}", n_components)
    if only_one_variable:
        features = [features[0]]*n_components

    def H_j(j, n_qubits):
        I_tensor = Pauli("I" * n_qubits)
        X_j = I_tensor.copy()
        X_j[j] = Pauli("X")
        X_jp1 = I_tensor.copy()
        X_jp1[j + 1] = Pauli("X")

        Y_j = I_tensor.copy()
        Y_j[j] = Pauli("Y")
        Y_jp1 = I_tensor.copy()
        Y_jp1[j + 1] = Pauli("Y")

        Z_j = I_tensor.copy()
        Z_j[j] = Pauli("Z")
        Z_jp1 = I_tensor.copy()
        Z_jp1[j + 1] = Pauli("Z")
        return SparsePauliOp(X_j@X_jp1) + SparsePauliOp(Y_j@Y_jp1) + SparsePauliOp(Z_j@Z_jp1)

    evolve_block = [PauliEvolutionGate(H_j(j, num_qubits), time=features[j]*evolve_time_t/trotter_time_T) for j in range(n_components)]
    
    circuit = QuantumCircuit(num_qubits)


    initial_state = [1/np.sqrt(2), -1/np.sqrt(2)]
    if inverse:
        for _ in range(trotter_time_T):
            for evo in evolve_block:
                circuit.append(evo, range(num_qubits))
        return circuit.inverse()

    random_haar_states = [random_statevector(2, seed = seed) for seed in range(num_qubits)]
    for i in range(num_qubits):
        circuit.initialize(random_haar_states[i], i)

    for _ in range(trotter_time_T):
        for evo in evolve_block:
            circuit.append(evo, range(num_qubits))
    return circuit

   
    
def Hamiltonian_time_evolution_encoding(n_components, trotter_time_T, evolve_time_t, inverse = False):
    return QiskitEncodingCircuit(Hamiltonian_time_evolution_encoding_qiskit(n_components, trotter_time_T, evolve_time_t, inverse))



def FQK_kernel_circuit(encoding_circuit, num_qubits, **kwargs):
    U = encoding_circuit(inverse = False, num_qubits = num_qubits, **kwargs)
    U_inv = encoding_circuit(inverse = True, num_qubits = num_qubits, **kwargs)

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(U, inplace = True)
    circuit.compose(U_inv, inplace = True)
    return circuit


def ChebyshevTowerAndHEE(num_qubits, num_features, num_layers):
    circuit =  ChebyshevTower(num_qubits=num_qubits, num_features=num_features, num_chebyshev=num_qubits, alpha = 2, hadamard_start=False)
    circuit += HEE_rzrxrz(num_qubits, num_features, num_layers)
    return circuit

def SimpleAnalyticalCircuit_qiskit(num_qubits, num_layers):
    """
    SimpleAnalyticalCircuit(num_qubits, num_layers)
    Returns a circuit that is similar to the one used in IQP.
    """
    #Rx(x)Ry(p1)Ry(p2)
    QC = QuantumCircuit(num_qubits)

    features = ParameterVector(f"x", 1)
    parameters = ParameterVector(f"p", num_layers)

    QC.rx(features[0], 0)
    for i in range(0, num_layers):
        QC.rx(parameters[i], 0)
    
    return QC

def SimpleAnalyticalCircuit(num_features, num_qubits, num_layers):
    return QiskitEncodingCircuit(SimpleAnalyticalCircuit_qiskit(num_qubits, num_layers))
    

circuits_dictionary = {
    "IQPLikeCircuit": IQPLikeCircuit,
    "Separable_rx": Separable_rx,
    "HardwareEfficientEmbeddingCircuit": HardwareEfficientEmbeddingCircuit,
    "Hamiltonian_time_evolution_encoding": Hamiltonian_time_evolution_encoding, 
    "NoCircuit": "NoCircuit",
    "SimpleAnalyticalCircuit": SimpleAnalyticalCircuit
}


circuits_dictionary_qiskit = {
    "IQPLikeCircuit": IQPLikeCircuit_qiskit,
    "Separable_rx": Separable_rx_qiskit,
    "Separable_rx_qiskit": Separable_rx_qiskit,
    "HardwareEfficientEmbeddingCircuit": HardwareEfficientEmbeddingCircuit_qiskit,
    "HardwareEfficientEmbeddingCircuit_qiskit": HardwareEfficientEmbeddingCircuit_qiskit,
    "Hamiltonian_time_evolution_encoding": Hamiltonian_time_evolution_encoding_qiskit,
    "NoCircuit": "NoCircuit",
    "YZ_CX_EncodingCircuit": YZ_CX_EncodingCircuit, 
    "MultiControlEncodingCircuit": MultiControlEncodingCircuit,
    "ChebyshevPQC": ChebyshevPQC, 
    "ChebyshevRx": ChebyshevRx,
    "ChebyshevTowerAndHEE": ChebyshevTowerAndHEE,
    "SimpleAnalyticalCircuit": SimpleAnalyticalCircuit
}
