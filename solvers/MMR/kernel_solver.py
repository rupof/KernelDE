from squlearn.encoding_circuit.encoding_circuit_derivatives import *
from squlearn.util import Executor
from qiskit.primitives import Estimator, Sampler
from squlearn.kernel.matrix import FidelityKernel, ProjectedQuantumKernel
from scipy.optimize import minimize
from squlearn.encoding_circuit import *
import numpy as np
from squlearn.observables import *
from circuits.circuits import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from DE_Library.qnn_and_kernels_wrappers import get_differentials

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
            kernel_tensor (tuple): A tuple containing derivatives of kernel objects for K_(0) and K_(1), ... , K_(n).
            lamb (float, optional): Parameter for the g function. Defaults to 20.
            k (float, optional): Parameter for the g function. Defaults to 0.1.
            regularization_parameter (float, optional): Regularization parameter for the loss function. Defaults to 1.
            f_initial (float, optional): Initial value of the dependent variable. Defaults to 0.0.
        """
        self.kernel_tensor = kernel_tensor
        self.regularization_parameter = regularization_parameter

    def f_alpha_order(self, alpha_, kernel_tensor, order):
        """Calculates f_alpha.

        Args:
            alpha_ (np.ndarray): The vector of alphas, of shape (len(x_span)+1, 1).
            kernel_tensor (tuple): A tuple containing kernel objects for f_alpha_0 and f_alpha_1.
            order (int): Order of the kernel.

        Returns:
            np.ndarray: The vector of f_alphas, of shape (len(x_span), 1).
        """
        alpha = alpha_[1:]
        if order == 0:
            return np.dot(kernel_tensor[order], alpha) + alpha_[0]
        return np.dot(kernel_tensor[order], alpha) 

    
    def loss_function(self, alpha_, L_functional, f_initial, x_span, kernel_tensor):
        """Calculates the loss function.

        Args:
            alpha_ (np.ndarray): The vector of alphas, of shape (len(x_span)+1, 
            L_functional (function): The L functional, that describes the argument of the loss function. For example: L_functional = dfdx(alpha, K_1) - g(f(x, alpha, K_0), x)
            
            Example:

            def L_g()
            
            #(self.f_alpha_1(alpha_, kernel_order_1) - g(self.f_alpha_0(alpha_, kernel_order_0), x_span)
        """
        f_alpha_tensor = np.array([self.f_alpha_order(alpha_, kernel_tensor, i) for i in range(len(kernel_tensor))])
        #print(f_alpha_tensor.shape, f_initial)
        sum1 = np.sum((L_functional(f_alpha_tensor, x_span)**2)) #Functional
        sum2 = np.sum((f_alpha_tensor[:,0][:len(f_initial)] - f_initial)**2) #Initial condition
        L = sum2 + sum1 * self.regularization_parameter

        return L
    
    def create_kernel_L_functional(self, L_functional):
        def kernel_L_functional(f_alpha_tensor, x_span):
            f_tensor = get_differentials(f_alpha_tensor, x_span)
            return L_functional(f_tensor)
        return kernel_L_functional
    
    def create_loss_function(self, L_functional, f_initial, x_span):
        def loss_function(alpha_):
            return self.loss_function(alpha_, L_functional, f_initial, x_span, self.kernel_tensor)
        return loss_function


    def solver(self, x_span, f_initial, L_functional):
        """Solves the differential equation using minimize from scipy.optimize.

        Args:
            x_span (np.ndarray): The span of the independent variable.
            kernel_tensor (tuple): A tuple containing kernel objects for f_alpha_0 and f_alpha_1.
            f_initial (np.ndarray): The initial value of the dependent variable.
            L_functional (function): The L functional, that describes the argument of the loss function. 
                L_functional is assumed to be a function that an argument f_tensor, where x, f, dfdx, dfdxdx = f_tensor.

                Example:

                Simple Harmonic Oscillator:

                def L_functional(f_tensor):
                    x = f_tensor[0]
                    f = f_tensor[1]
                    dfdx = f_tensor[2]
                    dfdxdx = f_tensor[3]
                    return dfdxdx + f

                

        Returns:
            tuple: A tuple containing the solution and optimal alpha.
        """

        alpha_0 = np.ones(len(x_span) + 1)

        _L_functional = self.create_kernel_L_functional(L_functional)



        L_loss_by_iteration = []
        f_by_iteration = []
        f_prime_by_iteration = []
        
        def store_loss(x):
            L_loss_by_iteration.append(self.loss_function(x, _L_functional, f_initial, x_span, self.kernel_tensor))
            f_by_iteration.append(self.f_alpha_order(x, self.kernel_tensor, 0))
            f_prime_by_iteration.append(self.f_alpha_order(x, self.kernel_tensor, 1))

        loss_function_ = self.create_loss_function(_L_functional, f_initial, x_span)
        
        result = minimize(loss_function_, alpha_0, 
            options={'disp': False, 'maxiter': 10000}, callback=store_loss)
        optimal_alpha = result.x
        solution = self.f_alpha_order(optimal_alpha, self.kernel_tensor, 0)
        return [solution, optimal_alpha], [L_loss_by_iteration, f_by_iteration, f_prime_by_iteration]

    
