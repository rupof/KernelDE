from squlearn.encoding_circuit.encoding_circuit_derivatives import *
from squlearn.util import Executor
from qiskit.primitives import Estimator, Sampler
from squlearn.kernel.matrix import FidelityKernel, ProjectedQuantumKernel
from scipy.optimize import minimize
from squlearn.encoding_circuit import *
import numpy as np
from squlearn.observables import *
from circuits.circuits import *
from squlearn.qnn.lowlevel_qnn import LowLevelQNN
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from .kernel_solver import *


import numpy as np
from scipy.optimize import minimize

cache = {}

class FQK_solver:

    def __init__(self, circuit_information, executor, regularization_parameter=1, CircuitInformation = None):
        """
        Initializes the Solver object.

        Args:
            circuit_information (tuple): A tuple containing the encoding circuit and the number of qubits.
            regularization_parameter (float, optional): Regularization parameter for the loss function. Defaults to 1.
        """

        self.encoding_circuit, self.num_qubits = circuit_information["encoding_circuit"], circuit_information["num_qubits"]
        self.executor = executor
        self.circuit_information_complete = circuit_information
        self.circuit_information = {key: value for key, value in circuit_information.items() if key not in {"encoding_circuit", "num_qubits"}}
        self.regularization_parameter = regularization_parameter    
        self.CircuitInformation = CircuitInformation
    
    def get_plotting_relevant_info(self):
        info =  self.circuit_information_complete
        if "encoding_circuit" in info:
            info["encoding_circuit"] = self.encoding_circuit.__name__
        return {**info }
    def print_plotting_relevant_info(self):
        info = self.get_plotting_relevant_info()
        text = "FQK: "
        for key, value in info.items():
            if key == "encoding_circuit":
                text += f"{value}, "
            elif key == "envelope":
                text += f"{value} "
            else:
                text += f"{key}: {value}, "
        return text
    

    @staticmethod
    def x_to_circuit_format(x):
        """
        x: np.array of shape (n, m) where n is the number of samples and m is the number of features.

        Convert the input data to the format that the program can accept.
        return: np.array of shape (n*n, 2*m) where n is the number of samples and m is the number of features.

        Note, this is necessary because of the way the quantum circuit for FQK is constructed.
        Example: 
        Take x = np.array([[1], 
                            [2], 
                            [3]])
        
        Then, the FQK circuit implements U(x) U^dag(x') |0> = |f(x, x')> |0> 

        Thus, we need to artificially create the input data with all possible x combinations in the format of x and x'.
        """    
        n, m = x.shape
        x_list_circuit_format = np.zeros((n*n, 2*m))

        # Use broadcasting to concatenate the features directly
        x_list_circuit_format[:, :m] = x.repeat(n, axis=0)
        x_list_circuit_format[:, m:] = np.tile(x, (n, 1))
        
        return x_list_circuit_format
    
    @staticmethod
    def P0_squlearn(num_qubits):
        """
        Create the P0 observable: (|0><0|)^\otimes n for the quantum circuit in the format of the squlearn library. 
        Note that |0><0| = 0.5*(I + Z) 

        Parameters:
        num_qubits: int, the number of qubits in the quantum circuit.

        return:
        - CustomObservable: The P0 observable in the format of the squlearn library.
        - coefficients: The coefficients of the P0 observable to be used in the QNN squlearn evaluation

        """
        from qiskit.quantum_info import SparsePauliOp
        
        P0_single_qubit = SparsePauliOp.from_list([("Z", 0.5), ("I", 0.5)])
        P0_temp = P0_single_qubit
        for i in range(1, num_qubits):
            P0_temp = P0_temp.expand(P0_single_qubit)
        observable_tuple_list = P0_temp.to_list()
        pauli_str = [observable[0] for observable in observable_tuple_list]    
        coefficients = [np.real(observable[1]) for observable in observable_tuple_list]
        return CustomObservable(num_qubits, pauli_str, parameterized=True), coefficients
    

    def FQK_QNN(self):
        """
        Create the FQK QNN for the given encoding circuit and the executor.

        Parameters:
        - EncodingCircuit: The encoding circuit to be used. EncodingCircuit should be in qiskit format, not squlearn format!
        - num_qubits: The number of qubits in the quantum circuit.
        - executor: The executor to be used.
        - **kwargs: The additional parameters to be passed to the encoding circuit.

        Returns:
        - qnn_: The FQK QNN.
        - P0_coef: The coefficients of the P0 observable to be used in the QNN squlearn evaluation
        """
        FQK_Circuit = QiskitEncodingCircuit(FQK_kernel_circuit(self.encoding_circuit(num_qubits = self.num_qubits, num_features = 1, **self.circuit_information) ), feature_label=["x", "y"]) 
        #Create P0 observable
        P0_, P0_coef = self.P0_squlearn(self.num_qubits)
        qnn_ = LowLevelQNN(FQK_Circuit, P0_, self.executor)
        return qnn_, P0_coef
    
    def get_FQK_kernel_derivatives(self, x_array, qnn_, coef, f_initial, **kwargs):
        """
        Get the FQK kernel and its derivatives for the given input data.

        Parameters:
        - x_array: The input data. np.array of shape (n, m) where n is the number of samples and m is the number of features.
        - qnn_: The FQK QNN.
        - coef: The coefficients of the P0 observable to be used in the QNN squlearn evaluation

        Returns:
        - output_f: The FQK kernel.
        - output_dfdx: The derivatives of the FQK kernel:  shape (n, n, m*2),  the last dimension is the derivative with respect to the input data. 
        """
        if self.CircuitInformation in cache:
            return  cache[self.CircuitInformation]
        else:
            print("New calculation FQK for qubits:", self.CircuitInformation.get_info()["num_qubits"], self.CircuitInformation.get_info())
            x_array = x_array.reshape(-1, 1) #reshape to column vector
            x_list_circuit_format = self.x_to_circuit_format(x_array)

            if qnn_.num_parameters != 0:
                np.random.seed(1)
                params = np.random.rand(qnn_.num_parameters)
            else:
                params = []
            output_f = qnn_.evaluate(x_list_circuit_format, params, coef, "f")["f"]  # (n*n, )
            output_dfdx = qnn_.evaluate(x_list_circuit_format, params, coef, "dfdx")["dfdx"] # (n*n, 2*m)

            if len(f_initial) == 2:
                output_dfdxdx = qnn_.evaluate(x_list_circuit_format, params, coef, "dfdxdx")["dfdxdx"] # (n*n, 
                output_dfdxdx = output_dfdxdx.reshape((len(x_array), len(x_array), len(x_array[0])*2, len(x_array[0])*2))

            else:
                output_dfdxdx = np.zeros((len(x_array), len(x_array), len(x_array[0])*2, len(x_array[0])*2))


            #reshape the output to the shape of the gram matrix
            output_f = output_f.reshape((len(x_array), len(x_array)))
            output_dfdx = output_dfdx.reshape((len(x_array), len(x_array), len(x_array[0])*2))


            cache[self.CircuitInformation] = output_f, output_dfdx[:,:,0], output_dfdxdx[:,:,0,0]
            return output_f, output_dfdx[:,:,0], output_dfdxdx[:,:,0,0]

   
    def solver(self, x_span, f_initial, L_functional, return_derivatives = False):
        """
        Solve the ODE using the PQK solver.

        First, the PQK QNN is created using the given quantum circuit. Then, the kernel and its derivatives are calculated using the QNN. Finally,
        the ODE is solved using the PQK solver. i.e the loss function is minimized to find the optimal alpha.

        Parameters:
        - Circuit_qiskit: The quantum circuit to be used.
        - num_points: The number of points to be used in the numerical solution.
        - num_qubits: The number of qubits in the quantum circuit.
        - sigma: The sigma parameter to be used in the RBF kernel.

        Returns:
        - FQK_sol: The solution using the FQK solver. (f, alpha)
        - kernel_list: The kernel list used in the solver if return_derivatives is True.
        """

        ### PQK

        FQK_qnn, obs_coef = self.FQK_QNN()
        K_f, K_dfdx, K_dfdxdx = self.get_FQK_kernel_derivatives(x_span, FQK_qnn, obs_coef, f_initial)
        kernel_list = [K_f, K_dfdx, K_dfdxdx]
        Solver_ = Solver(kernel_list, self.regularization_parameter)
        solution_ = Solver_.solver(x_span, f_initial, L_functional = L_functional)
        if return_derivatives:
            return solution_, kernel_list
        else:
            return solution_
    
    def get_Kernel(self, x_span):
        """
        Get the FQK kernel for the given input data.

        Parameters:
        - x_span: The input data. np.array of shape (n, m) where n is the number of samples and m is the number of features.

        Returns:
        - output_f: The FQK kernel.
        """
        FQK_qnn, obs_coef = self.FQK_QNN()
        K_f, _ = self.get_FQK_kernel_derivatives(x_span, FQK_qnn, obs_coef)
        return K_f
    
    