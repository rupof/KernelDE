from squlearn.encoding_circuit.encoding_circuit_derivatives import *
from squlearn.util import Executor
from qiskit.primitives import Estimator, Sampler
from squlearn.kernel.matrix import FidelityKernel, ProjectedQuantumKernel
from circuits.circuits import circuits_dictionary
from scipy.optimize import minimize

import numpy as np

print(circuits_dictionary)

class DifferentialEquationSolver:
    def __init__(self, x_initial, f_initial, x_span, kernel_method_info):
        """
        Initialize the solver.

        Parameters:
        - equation: A function representing the ordinary differential equation.
        - initial_conditions: Initial values for the dependent variables.
        - x_span: Time span for the solution (e.g., (t0, t_final)).
        - method: Numerical integration method (default is 'RK45').
        """
        self.x_initial = x_initial
        self.f_initial = f_initial
        self.x_span = x_span
        self.kernel_method_info = kernel_method_info
        self.encoding_circuit_label = self.kernel_method_info["encoding_circuit_label"]
        #self.encoding_circuit = circuits_dictionary[self.encoding_circuit_label]


    def get_kernel_object(self, encoding_circuit, executor_type, method, num_shots = None):
        if executor_type == "statevector":
            if method == "FQK":
                executor = Executor(Sampler(), shots=None, primitive_seed=1)
                Kernel = FidelityKernel(encoding_circuit, executor=executor)
            elif method == "PQK":
                executor = Executor(Estimator(), shots=None, primitive_seed=1)
                #Only for num_qubits < 8
                Kernel = ProjectedQuantumKernel(encoding_circuit, executor=executor)
        elif executor_type == "shots":
            if method == "FQK":
                executor = Executor("qasm_simulator",shots=num_shots, primitive_seed=1)
                Kernel = FidelityKernel(encoding_circuit, executor=executor)
            elif method == "PQK":
                executor = Executor(Estimator(), shots=num_shots, primitive_seed=1)
                Kernel = ProjectedQuantumKernel(encoding_circuit, executor=executor)
        return Kernel
    
    def get_explicit_circuit_and_derivatives(self, encoding_circuit, num_qubits, num_layers, **kwargs):
        """
        Parameters:
        - encoding_circuit: The encoding circuit to be used.
        - num_qubits: The number of qubits in the encoding circuit.
        - num_layers: The number of layers in the encoding circuit.
        - **kwargs: Additional arguments to be passed to the encoding circuit.
        Returns:
        - explicit_circuit_list: A list of the explicit circuits and their derivatives.
        [explicit_circuit, explicit_circuit_1st_derivative, explicit_circuit_2nd_derivative, ...]
        """
        derivative_order = 2
        
        explicit_circuit = encoding_circuit(num_qubits, num_layers, **kwargs)
        explicit_circuit_list = [explicit_circuit]

        explicit_circuit_derivative_object = EncodingCircuitDerivatives(explicit_circuit)
        for i in range(1, derivative_order):
            print("Calculating the {}th derivative of the circuit".format(i))
            explicit_circuit_ith_derivative = explicit_circuit_derivative_object.get_derivative("dx")
            explicit_circuit_list.append(explicit_circuit_ith_derivative)
        
        return explicit_circuit_list

    def evaluate_kernel(self, x_span, Kernel_circuit_list):
        """
        Parameters:
        - x_span: Time span for the solution (e.g., (t0, t_final)).
        - explicit_circuit_list: A list of the explicit circuits and their derivatives.
        [explicit_circuit, explicit_circuit_1st_derivative, explicit_circuit_2nd_derivative, ...]
        Returns:
        - kernel: The kernel matrix. [kernel, kernel_1st_derivative, kernel_2nd_derivative, ...]

        """

        if self.kernel_method_info["method"] == "pre-calculated":
            return self.kernel_method_info["kernel"]
        else: #else, we need to calculate the kernel
            kernel_tensor = np.zeros((len(Kernel_circuit_list), len(x_span), len(x_span)))    
            #Kernel_tensor[i] is the kernel matrix of the ith explicit circuit
            for i, Kernel in enumerate(Kernel_circuit_list):
                kernel_tensor[i] = Kernel.evaluate(x_span, x_span)
            return kernel_tensor
    
    def f_alpha_0(self, alpha_, kernel_order_0):
        """

        Parameters:
        - alpha: The vector of alphas, of shape (len(x_span)+1, 1).
        - kernel_tensor: a
        - initial_conditions: Initial values for the dependent variables.
        Returns:
        - f_alphas: The vector of f_alphas, of shape (len(x_span), 1).
        """
        b = alpha_[0]
        alpha = alpha_[1:]
        return b + kernel_order_0 @ alpha

    def f_alpha_1(self, alpha_, kernel_order_1):
        """
        Parameters:
        - alpha: The vector of alphas, of shape (len(x_span)+1, 1).
        - kernel_tensor: a
        - initial_conditions: Initial values for the dependent variables.
        Returns:
        - f_alphas: The vector of f_alphas, of shape (len(x_span), 1). of the first order derivative
        """
        alpha = alpha_[1:]
        return  kernel_order_1 @ alpha
    
    def g(self, f, x):
        lamb  = 20
        k = 0.1
        return -lamb*np.exp(-lamb*x*k)*np.sin(lamb*x) - lamb*k*f

    def loss_function(self, alpha, f_initial, x_span, kernel_order_0, kernel_order_1):
    
        sum1 = np.sum((self.f_alpha_1(alpha, kernel_order_1)-self.g(self.f_alpha_0(alpha, kernel_order_0), x_span))**2)
        sum2 = (self.f_alpha_0(alpha, kernel_order_0)[0]-f_initial)**2
        L = sum1 + sum2
        print(L)
        return L


    def solve(self, num_qubits = None, num_layers = None, **kwargs):
        """
        Solve the differential equation using the specified method.

        Returns:
        - solution: The solution of the differential equation.
        """
        # Get the kernel object
        kernel_list = []
        if self.kernel_method_info["method"] == "pre-calculated":
            kernel_tensor = self.kernel_method_info["kernel"]
        elif self.kernel_method_info["method"] != "pre-calculated":
            for explicit_circuit in explicit_circuit_list:
                print("Generating Circuit List and derivatives")
                explicit_circuit_list = self.get_explicit_circuit_and_derivatives(self.encoding_circuit, num_qubits, num_layers, **kwargs)
                print(len(explicit_circuit_list))

                print("Getting kernel object {}".format(explicit_circuit))
                kernel_list.append(self.get_kernel_object(explicit_circuit, 
                                            self.kernel_method_info["executor_type"], 
                                            self.kernel_method_info["method"], 
                                            self.kernel_method_info["num_shots"]))
                print("done")
            
            
            print("Evaluating Kernel")
            kernel_tensor = self.evaluate_kernel(self.x_span, kernel_list)
        print("Solving the differential equation")

        result = minimize(self.loss_function, np.ones(len(self.x_span)+1), 
                 args = (self.f_initial, self.x_span, kernel_tensor[0], kernel_tensor[1]),
                 options={'disp': True, 'maxiter':10000})
        optimal_alpha = result.x #optimal_alpha = [b, alpha_1, alpha_2, ...]
        # Evaluate the solution
        solution = self.f_alpha_0(optimal_alpha[:], kernel_tensor[0])
        
        
        return solution, optimal_alpha



