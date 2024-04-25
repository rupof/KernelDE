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

from utils.rbf_kernel_tools import *

class PQK_solver:

    def __init__(self, circuit_information, executor, envelope, regularization_parameter=1):
        """
        Initializes the Solver object.

        Args:
            circuit_information (tuple): A tuple containing the encoding circuit and the number of qubits.
            regularization_parameter (float, optional): Regularization parameter for the loss function. Defaults to 1.
        """

        self.encoding_circuit, self.num_qubits = circuit_information["encoding_circuit"], circuit_information["num_qubits"]
        self.executor = executor
        self.circuit_information = {key: value for key, value in circuit_information.items() if key not in {"encoding_circuit", "num_qubits"}}
        self.regularization_parameter = regularization_parameter
        self.envelope = envelope["function"]
        self.analytical_derivative = envelope["derivative_function"]
        self.analytical_derivative_2 = envelope["second_derivative_function"]
        self.envelope_parameters = {key: value for key, value in envelope.items() if key not in {"function", "derivative_function", "second_derivative_function"}}
    
    
    def PQK_QNN(self):
        """
        Create the PQK QNN for the given encoding circuit and the executor.

        Parameters:
        - EncodingCircuit: The encoding circuit to be used. EncodingCircuit should be in qiskit format, not squlearn format!
        - num_qubits: The number of qubits in the quantum circuit.
        - executor: The executor to be used.
        - **kwargs: The additional parameters to be passed to the encoding circuit.

        Returns:
        - qnn_: The PQK QNN.
        - observable_coef: The coefficients of the observable to be used in the QNN squlearn evaluation

        """
        EncodingCircuit = self.encoding_circuit
        num_qubits = self.num_qubits
        executor = self.executor


        PQK_Circuit = QiskitEncodingCircuit(EncodingCircuit(num_qubits = num_qubits, only_one_variable = True, **self.circuit_information))
        observable = SummedPaulis(num_qubits, op_str="XYZ", include_identity=False, full_sum=False) #i.e Summed Paulis for a 2 qubit system: IX, XI, IY, YI, IZ, ZI
        observable.get_pauli_mapped([1 for i in range(observable.num_parameters)]) #
        observable_coef = [1 for i in range(observable.num_parameters)]
        qnn_ = LowLevelQNN(PQK_Circuit, observable, self.executor)
        return qnn_, observable_coef
    
    def get_PQK_kernel_derivatives(self, x_array, qnn_, coef, **kwargs):
        """
        Get the PQK kernel and its derivatives for the given input data.

        Parameters:
        - x_array: The input data. np.array of shape (n) where n is the number of samples. m dimensional input is not supported for PQK yet.
        - qnn_: The PQK QNN.
        - coef: The coefficients of the observable to be used in the QNN squlearn evaluation
        - envelope: The envelope function to be used.
        - analytical_derivative: The analytical derivative function to be used.
        - **kwargs: The additional parameters to be passed to the envelope and analytical_derivative functions.

        Returns:
        - output_f_gramm_matrix: The PQK kernel.
        - output_dfdx_gramm_matrix: The derivatives of the PQK kernel. 
        """


        x_list_circuit_format = x_array
        output_f_column = qnn_.evaluate(x_list_circuit_format, [], coef, "f")["f"] # #shape (n,)    
        output_dfdx_qnn_column = qnn_.evaluate(x_list_circuit_format, [], coef, "dfdx")["dfdx"][:,0] #shape (n, 1)
        output_dfdxdx_qnn_column = qnn_.evaluate(x_list_circuit_format, [], coef, "dfdxdx")["dfdxdx"][:,0,0] #shape (n, 1, 1)

        #reshape the output_f_column, output_dfdx_qnn_column, output_dfdxdx_qnn_column to (n, n)
        #output_f_column = output_f_column.reshape(-1, 1) #reshape to column vector    
        print("shapes")
        print("output_f_gramm_matrix", output_f_column.shape)
        print("output_dfdx_gramm_matrix", output_dfdx_qnn_column.shape)
        print("output_dfdxdx_gramm_matrix", output_dfdxdx_qnn_column.shape)

        output_f_gramm_matrix = self.envelope(output_f_column, output_f_column, **kwargs).reshape(len(x_array), len(x_array)) #shape (n, n) #to be checked
        output_dfdx_gramm_matrix = self.analytical_derivative(output_f_column, output_f_column, **kwargs) * output_dfdx_qnn_column #shape (n, n) #to be checked

        output_dfdxdx_gramm_matrix = self.analytical_derivative_2(output_f_column, output_f_column, **kwargs) * output_dfdx_qnn_column + self.analytical_derivative(output_f_column, output_f_column, **kwargs) * output_dfdxdx_qnn_column #shape (n, n) #to be checked
        print("shapes")
        print("output_f_gramm_matrix", output_f_gramm_matrix.shape)
        print("output_dfdx_gramm_matrix", output_dfdx_gramm_matrix.shape)
        print("output_dfdxdx_gramm_matrix", output_dfdxdx_gramm_matrix.shape)

        return output_f_gramm_matrix, output_dfdx_gramm_matrix, output_dfdxdx_gramm_matrix[:,:]
    

    def solver(self, x_span, f_initial, L_functional):
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
        - PQK_sol: The solution using the PQK solver.
        - kernel_list: The kernel list.
        """

        ### PQK
        PQK_qnn, obs_coef = self.PQK_QNN()
        K_f, K_dfdx, K_dfdxdx = self.get_PQK_kernel_derivatives(x_span, PQK_qnn, obs_coef, **self.envelope_parameters)
        print("K_f", K_f.shape)
        print("K_dfdx", K_dfdx.shape)
        print("K_dfdxdx", K_dfdxdx.shape)
        kernel_list = np.array([K_f, K_dfdx, K_dfdxdx])
        Solver_ = Solver(kernel_list, self.regularization_parameter)
        solution_ = Solver_.solver(x_span, f_initial, L_functional = L_functional)
        return solution_, kernel_list
    
    def get_Kernel(self, x_span):
        """
        Get the PQK kernel for the given input data.

        Parameters:
        - x_span: The input data. np.array of shape (n) where n is the number of samples. m dimensional input is not supported for PQK yet.

        Returns:
        - output_f_gramm_matrix: The PQK kernel.
        """
        PQK_qnn, obs_coef = self.PQK_QNN()
        K_f, _ = self.get_PQK_kernel_derivatives(x_span, PQK_qnn, obs_coef, **self.envelope_parameters)
        return K_f
    