from squlearn.encoding_circuit.encoding_circuit_derivatives import *
from squlearn.util import Executor
from qiskit.primitives import Estimator, Sampler
from squlearn.kernel.matrix import FidelityKernel, ProjectedQuantumKernel
from scipy.optimize import minimize
from squlearn.encoding_circuit import *
import numpy as np
from squlearn.observables import *
from circuits.circuits import *
from squlearn.qnn.qnn import QNN
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


import numpy as np
from scipy.optimize import minimize


class Solver:
    """
    Solves the differential equation using the specified method.

    Attributes:
        kernel_tensor (tuple): A tuple containing kernel objects for f_alpha_0 and f_alpha_1.
        g (function): The g function.
        regularization_parameter (float): Regularization parameter for the loss function.

    Methods:
        f_alpha_0(alpha_, kernel_order_0): Calculates f_alpha_0.
        f_alpha_1(alpha_, kernel_order_1): Calculates f_alpha_1.
        g(f, x): Calculates the g function.
        loss_function(alpha_, f_initial, regularization_parameter, x_span, kernel_order_0, kernel_order_1): Calculates the loss function.
        solve(x_span, kernel_tensor, f_initial): Solves the differential equation.
    """

    def __init__(self, kernel_tensor, regularization_parameter=1):
        """
        Initializes the Solver object.

        Args:
            kernel_order_0 (int): Order of the kernel for f_alpha_0.
            kernel_order_1 (int): Order of the kernel for f_alpha_1.
            lamb (float, optional): Parameter for the g function. Defaults to 20.
            k (float, optional): Parameter for the g function. Defaults to 0.1.
            regularization_parameter (float, optional): Regularization parameter for the loss function. Defaults to 1.
            f_initial (float, optional): Initial value of the dependent variable. Defaults to 0.0.
        """

        self.kernel_order_0 = kernel_tensor[0]
        self.kernel_order_1 = kernel_tensor[1]
        self.regularization_parameter = regularization_parameter

    def f_alpha_0(self, alpha_, kernel_order_0):
        """Calculates f_alpha_0.

        Args:
            alpha_ (np.ndarray): The vector of alphas, of shape (len(x_span)+1, 1).
            kernel_order_0 (int): Order of the kernel.

        Returns:
            np.ndarray: The vector of f_alphas, of shape (len(x_span), 1).
        """

        b = alpha_[0]
        alpha = alpha_[1:]
        return b + np.dot(kernel_order_0, alpha)

    def f_alpha_1(self, alpha_, kernel_order_1):
        """Calculates f_alpha_1.

        Args:
            alpha_ (np.ndarray): The vector of alphas, of shape (len(x_span)+1, 1).
            kernel_order_1 (int): Order of the kernel.

        Returns:
            np.ndarray: The vector of f_alphas, of shape (len(x_span), 1).
        """

        alpha = alpha_[1:]
        return np.dot(kernel_order_1, alpha)


    def loss_function(self, alpha_, g, f_initial, x_span, kernel_order_0, kernel_order_1):
        """Calculates the loss function.

        Args:
            alpha_ (np.ndarray): The vector of alphas, of shape (len(x_span)+1, 

        """
        sum1 = np.sum((self.f_alpha_1(alpha_, kernel_order_1) - g(self.f_alpha_0(alpha_, kernel_order_0), x_span))**2)
        sum2 = (self.f_alpha_0(alpha_, kernel_order_0)[0] - f_initial)**2
        L = sum2 + sum1 * self.regularization_parameter
        return L

    def solver(self, x_span, f_initial, g):
        """Solves the differential equation using minimize from scipy.optimize.

        Args:
            x_span (np.ndarray): The span of the independent variable.
            kernel_tensor (tuple): A tuple containing kernel objects for f_alpha_0 and f_alpha_1.

        Returns:
            tuple: A tuple containing the solution and optimal alpha.
        """

        kernel_order_0, kernel_order_1 = self.kernel_order_0, self.kernel_order_1
        regularization_parameter = self.regularization_parameter

        alpha_0 = np.ones(len(x_span) + 1)

        print("Initial loss: ", self.loss_function(alpha_0, g, f_initial, x_span, kernel_order_0, kernel_order_1))
        result = minimize(self.loss_function, alpha_0, args=(g,
            f_initial, x_span, kernel_order_0, kernel_order_1),
            options={'disp': True, 'maxiter': 10000})

        optimal_alpha = result.x
        solution = self.f_alpha_0(optimal_alpha, kernel_order_0)
        _ = 0
        return [solution, optimal_alpha], _

    

