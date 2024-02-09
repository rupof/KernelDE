from squlearn.encoding_circuit.encoding_circuit_derivatives import *
from squlearn.util import Executor
from qiskit.primitives import Estimator, Sampler
from squlearn.kernel.matrix import FidelityKernel, ProjectedQuantumKernel
from scipy.optimize import minimize
from squlearn.encoding_circuit import *
import numpy as np
from matplotlib import pyplot as plt

from squlearn.observables import *
from circuits.circuits import *
from squlearn.qnn.qnn import QNN

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint



class DifferentialEquationSolver:
    def __init__(self, x_initial, f_initial, x_span, kernel_method_info):
        self.x_initial = x_initial
        self.f_initial = f_initial
        self.x_span = x_span
        self.kernel_method_info = kernel_method_info

    def f_alpha_0(self, alpha, kernel_order_0):
        b = alpha[0]
        alpha_values = alpha[1:]
        return b + kernel_order_0 @ alpha_values

    def f_alpha_1(self, alpha, kernel_order_1):
        alpha_values = alpha[1:]
        return kernel_order_1 @ alpha_values

    def g(self, f, x):
        lamb = 20
        k = 0.1
        return -lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) - lamb * k * f

    def loss_function(self, alpha, regularization_parameter, kernel_order_0, kernel_order_1):
        alpha_values = alpha[1:]
        sum1 = np.sum((self.f_alpha_1(alpha, kernel_order_1) - self.g(self.f_alpha_0(alpha, kernel_order_0), self.x_span))**2)
        sum2 = (self.f_alpha_0(alpha, kernel_order_0)[0] - self.f_initial)**2
        L = sum2 + sum1 * regularization_parameter
        return L

    def solve(self):
        regularization_parameter = 1
        alpha_0 = np.ones(len(self.x_span) + 1)
        result = minimize(self.loss_function, alpha_0,
                          args=(regularization_parameter, self.kernel_method_info["kernel_tensor"][0],
                                self.kernel_method_info["kernel_tensor"][1]),
                          options={'disp': True, 'maxiter': 10000})

        optimal_alpha = result.x
        solution = self.f_alpha_0(optimal_alpha, self.kernel_method_info["kernel_tensor"][0])

        return solution, optimal_alpha
    
    @staticmethod
    def rbf_fun(x, y, sigma=1):
        return np.exp(-(x - y)**2 / (2 * sigma**2))

    @staticmethod
    def rbf_kernel_manual(x, y, sigma=1):
        kernel = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                kernel[i, j] = DifferentialEquationSolver.rbf_fun(x[i], y[j], sigma)
        return kernel

    @staticmethod
    def analytical_derivative_rbf_kernel(x, y, sigma=1):
        derivative = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                derivative[i, j] = -DifferentialEquationSolver.rbf_fun(x[i], y[j], sigma) * (2 * (x[i] - y[j]) / (2 * sigma**2))
        return derivative

    @staticmethod
    def rbf_fun(x, y, sigma=1):
        return np.exp(-(x - y)**2 / (2 * sigma**2))

    @staticmethod
    def rbf_kernel_manual(x, y, sigma=1):
        kernel = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                kernel[i, j] = DifferentialEquationSolver.rbf_fun(x[i], y[j], sigma)
        return kernel

    @staticmethod
    def analytical_derivative_rbf_kernel(x, y, sigma=1):
        derivative = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                derivative[i, j] = -DifferentialEquationSolver.rbf_fun(x[i], y[j], sigma) * (2 * (x[i] - y[j]) / (2 * sigma**2))
        return derivative
    

    @staticmethod
    def P0_squlearn(num_qubits):
        from qiskit.quantum_info import SparsePauliOp
        P0_single_qubit = SparsePauliOp.from_list([("Z", 0.5), ("I", 0.5)])
        P0_temp = P0_single_qubit
        for _ in range(1, num_qubits):
            P0_temp = P0_temp.expand(P0_single_qubit)
        observable_tuple_list = P0_temp.to_list()
        pauli_str = [observable[0] for observable in observable_tuple_list]
        coefficients = [observable[1] for observable in observable_tuple_list]
        return CustomObservable(num_qubits, pauli_str, parameterized=True), coefficients

    @staticmethod
    def x_to_circuit_format(x):
        n, m = x.shape
        x_list_circuit_format = np.zeros((n * n, 2 * m))
        x_list_circuit_format[:, :m] = x.repeat(n, axis=0)
        x_list_circuit_format[:, m:] = np.tile(x, (n, 1))
        return x_list_circuit_format

    @staticmethod
    def FQK_QNN(EncodingCircuit, num_qubits, executor, **kwargs):
        FQK_Circuit = QiskitEncodingCircuit(FQK_kernel_circuit(EncodingCircuit, num_qubits, only_one_variable=True, **kwargs))
        P0_, P0_coef = DifferentialEquationSolver.P0_squlearn(num_qubits)
        qnn_ = QNN(FQK_Circuit, P0_, executor, result_caching=False, optree_caching=False)
        return qnn_, P0_coef

    @staticmethod
    def get_FQK_kernel_derivatives(x_array, qnn_, coef):
        x_list_circuit_format = DifferentialEquationSolver.x_to_circuit_format(x_array)
        output_f = qnn_.evaluate("f", x_list_circuit_format, [], coef)["f"]
        output_dfdx = qnn_.evaluate("dfdx", x_list_circuit_format, [], coef)["dfdx"]
        output_f = output_f.reshape((len(x_array), len(x_array)))
        output_dfdx = output_dfdx.reshape((len(x_array), len(x_array), len(x_array[0]) * 2))
        return output_f, output_dfdx

    def FQK_solver(self, num_points):
        x_span = np.linspace(0, 1, num_points)
        FQK_qnn, P0_coef = self.FQK_QNN(self.kernel_method_info["encoding_circuit_label"],
                                        self.kernel_method_info["encoding_circuit_parameters"]["num_qubits"],
                                        Executor(self.kernel_method_info["executor_type"]),
                                        num_layers=2, rotation_gate=self.kernel_method_info["encoding_circuit_parameters"]["rotation_gate"])

        K_f, K_dfdx = self.get_FQK_kernel_derivatives(x_span.reshape(-1, 1), FQK_qnn, P0_coef)
        kernel_list = np.array([K_f, K_dfdx[:, :, 0]])
        solution, optimal_alpha = self.solve(x_span, kernel_list, self.loss_function, self.f_initial)
        FQK_sol = solution
        return FQK_sol, kernel_list

    def PQK_solver(self, num_points, sigma):
        x_span = np.linspace(0, 1, num_points)
        PQK_qnn, obs_coef = self.PQK_QNN(self.kernel_method_info["encoding_circuit_label"],
                                          self.kernel_method_info["encoding_circuit_parameters"]["num_qubits"],
                                          Executor(self.kernel_method_info["executor_type"]),
                                          num_layers=2, rotation_gate=self.kernel_method_info["encoding_circuit_parameters"]["rotation_gate"])

        K_f, K_dfdx = self.get_PQK_kernel_derivatives(x_span, PQK_qnn, obs_coef,
                                                      rbf_kernel_manual, analytical_derivative_rbf_kernel, sigma=sigma)
        kernel_list = np.array([K_f, K_dfdx])
        solution, optimal_alpha = self.solve(x_span, kernel_list, self.loss_function, self.f_initial)
        PQK_sol = solution
        return PQK_sol, kernel_list
