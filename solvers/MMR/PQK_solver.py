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
        self.circuit_information_complete = circuit_information
        self.circuit_information = {key: value for key, value in circuit_information.items() if key not in {"encoding_circuit", "num_qubits"}}
        self.regularization_parameter = regularization_parameter
        self.envelope = envelope["function"]
        self.analytical_derivative = envelope["derivative_function"]
        self.analytical_derivative_2 = envelope["second_derivative_function"]
        self.envelope_parameters = {key: value for key, value in envelope.items() if key not in {"function", "derivative_function", "second_derivative_function"}}
    
    def get_plotting_relevant_info(self):
        info =  self.circuit_information_complete
        if "encoding_circuit" in info:
            info["encoding_circuit"] = self.encoding_circuit.__name__
        info["envelope"] = self.envelope.__name__
        return {**info, **self.envelope_parameters}
    def print_plotting_relevant_info(self):
        info = self.get_plotting_relevant_info()
        text = "PQK: "
        for key, value in info.items():
            if key == "encoding_circuit":
                text += f"{value}, "
            elif key == "envelope":
                pass
                #text += f"{value} "
            else:
                text += f"{key}: {value},"
        return text
    
    def PQK_observable(self, measurement = "XYZ"):
        """"
        Returns the observable for the PQK solver

        Args:

        num_qubits (int): number of qubits in the system
        measurement (str): measurement operator to be applied to the qubits (default: "XYZ")

        Returns:
        _measurement (list): list of SinglePauli objects representing the measurement operator (shape: num_qubits*len(measurement))

        """
        num_qubits = self.num_qubits
        if isinstance(measurement, str):
                    _measurement = []
                    for m_str in measurement:
                        if m_str not in ("X", "Y", "Z"):
                            raise ValueError("Unknown measurement operator: {}".format(m_str))
                        for i in range(num_qubits):
                            _measurement.append(SinglePauli(num_qubits, i, op_str=m_str))
        return _measurement
    
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
        executor = self.executor

        try:
            PQK_Circuit = PQK_kernel_wrapper(QiskitEncodingCircuit(EncodingCircuit(num_qubits = self.num_qubits, num_features = 1, **self.circuit_information)))
        except:
            PQK_Circuit = EncodingCircuit(num_qubits = self.num_qubits, num_features = 1, **self.circuit_information)
        qnn_ = LowLevelQNN(PQK_Circuit, self.PQK_observable("XYZ"), self.executor)
        return qnn_
    
    def get_PQK_kernel_derivatives(self, x_array, qnn_, coef, f_initial, **kwargs):
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
        if qnn_.num_parameters != 0:
            np.random.seed(1)
            params = np.random.rand(qnn_.num_parameters)
        else:
            params = []

        O = qnn_.evaluate(x_list_circuit_format, params, coef, "f")["f"] # #shape (n,num_qubits*len(measurement))
        dOdx = qnn_.evaluate(x_list_circuit_format, params, coef, "dfdx")["dfdx"] #shape (n,num_qubits*len(measurement), 1)

        K_envelope = self.envelope(O, O, **kwargs) #shape (n, n) 
        K_envelope_dx = np.einsum("njl->nj", self.analytical_derivative(O, O, **kwargs) * dOdx[:,:,0]) #shape (n, num_qubits*len(measurement), 1)
        #
        if len(f_initial) == 2:
            dOdxdx = qnn_.evaluate(x_list_circuit_format, params, coef, "dfdxdx")["dfdxdx"] #shape (n, num_qubits*len(measurement), 1, 1)
            K_envelope_dxdx = np.einsum("njl->nj", self.analytical_derivative_2(O, O, **kwargs) * dOdx[:,:,0] * dOdx[:,:,0])   
            +  np.einsum("njl->nj", self.analytical_derivative(O, O, **kwargs) * dOdxdx[:,:,0,0]) #shape (n, num_qubits*len(measurement), 1)
            
        else:
            print("zero")
            K_envelope_dxdx = np.zeros_like(K_envelope_dx)
       
        return K_envelope, K_envelope_dx, K_envelope_dxdx
    

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
        - PQK_sol: The solution using the PQK solver (f, alpha).
        - kernel_list: The kernel list used in the solver if return_derivatives is True.
        """

        ### PQK
        PQK_qnn = self.PQK_QNN()
        obs_coef = []
        K_f, K_dfdx, K_dfdxdx = self.get_PQK_kernel_derivatives(x_span, PQK_qnn, obs_coef, f_initial, **self.envelope_parameters)
        print("K_f", K_f.shape)
        print("K_dfdx", K_dfdx.shape)
        print("K_dfdxdx", K_dfdxdx.shape)
        kernel_list = np.array([K_f, K_dfdx, K_dfdxdx])
        Solver_ = Solver(kernel_list, self.regularization_parameter)
        solution_ = Solver_.solver(x_span, f_initial, L_functional = L_functional)
        if return_derivatives:
            return solution_, kernel_list
        else:
            return solution_
    
    
    def get_Kernel(self, x_span):
        """
        Get the PQK kernel for the given input data.

        Parameters:
        - x_span: The input data. np.array of shape (n) where n is the number of samples. m dimensional input is not supported for PQK yet.

        Returns:
        - output_f_gramm_matrix: The PQK kernel.
        """
        PQK_qnn = self.PQK_QNN()
        obs_coef = []
        K_f, _ = self.get_PQK_kernel_derivatives(x_span, PQK_qnn, obs_coef, **self.envelope_parameters)
        return K_f
    