from squlearn.encoding_circuit.encoding_circuit_derivatives import *


import numpy as np
from scipy.integrate import odeint
from solvers.MMR.kernel_solver import Solver
from solvers.MMR.PQK_solver import PQK_solver
from solvers.MMR.FQK_solver import FQK_solver

from utils.rbf_kernel_tools import *
from circuits.circuits import *

import matplotlib.pyplot as plt


# General form of the differential equation is given by:
# f'(x) = g(f, x)


def g_paper(f, x):
        """
        df/dx = -lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) - lamb * k * f

        solution: f(x) = np.exp(-lamb * x * k) * np.cos(lamb * x), f(0) = 1
        """
        lamb = 20
        k = 0.1
        return -lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) - lamb * k * f

def g_exp(f, x):
    """
    df/dx = lamb * np.exp(f * k) 
    f(0.001) = np.log(0.001)

    solution: f(x) = np.log(x)
    """
    lamb = 1
    k = 1
    return np.exp(-f*k)*lamb

def g_exp_2(f, x):
    """
    df/dx = 2*f+4*cos(x)-8*sin(x), f(0) = 3

    solution: f(x) = 3*exp(2*x) + 4*sin(x)
    """
    return 2*f+4*np.cos(x)-8*np.sin(x)
####################################3


x_line = np.linspace(0.1, 1, 20)

f_initial = [np.log(x_line[0])]

######################3

g = g_exp

#Numerical and Analytical Solutions
f_odeint = odeint(g, f_initial, x_line[:])
#f_analytical_sol = f_analytical_fun(x_span)


def L_functional_1ODE(f_alpha_tensor, x_span):
    """
    L_functional = dfdx - g(f(x), x)
    """
    f = f_alpha_tensor[0]
    dfdx = f_alpha_tensor[1]

    return dfdx - g(f, x_span)
    

#Classical Solver
#RBF
RBF_kernel_list = [rbf_kernel_manual(x_line, x_line, sigma = 0.2), analytical_derivative_rbf_kernel(x_line, x_line, sigma = 0.2), analytical_derivative_rbf_kernel_2(x_line, x_line, sigma = 0.2)]
Solver_test = Solver(RBF_kernel_list, regularization_parameter=1)
#solution_RBF, _ = Solver_test.solver(x_line, f_initial, L_functional = L_functional_1ODE)
#f_RBF, optimal_alpha_RBF = solution_RBF[0], solution_RBF[1] #fix bug here


#PQK
sigma_list = np.linspace(0.1, 1, 70)
mse_list = np.zeros_like(sigma_list)
#for idx, sigma in enumerate(sigma_list):
sigma = 1.5
PQK_solver_test = PQK_solver({"encoding_circuit": Separable_rx_qiskit, 
                            "num_qubits": 1, #8
                            "num_layers": 2,
                            },
                            Executor("pennylane"), 
                            envelope={"function": rbf_kernel_manual, 
                                        "derivative_function": analytical_derivative_rbf_kernel, 
                                        "second_derivative_function": analytical_derivative_rbf_kernel_2,
                                        "sigma": sigma})


#solution_PQK, kernel_list_PQK = PQK_solver_test.solver(x_line, f_initial, L_functional = L_functional_1ODE)
#f_PQK, optimal_alpha_PQK = solution_PQK[0] ##fix bug here
#mse_list[idx] = np.mean((f_PQK - f_odeint)**2)




    # = solution_PQK[1]


FQK_solver_test = FQK_solver({"encoding_circuit": HardwareEfficientEmbeddingCircuit_qiskit, 
                              "num_qubits": 1, #6
                              "num_layers": 10,
                              },
                              Executor("pennylane"),)
solution_FQK, kernel_listFQK = FQK_solver_test.solver(x_line, f_initial, L_functional_1ODE)
f_FQK, optimal_alpha_FQK = solution_FQK[0]





x_span_plot = x_line.reshape(-1, 1)
plt.plot(x_span_plot, f_odeint, "-*",label="odeint")
#plt.plot(x_span_plot, f_RBF, "x", label="RBF")
#plt.plot(x_span_plot, f_PQK, label="PQK")
plt.plot(x_span_plot, f_FQK, "-x",label="FQK")
#plt.plot(x_span, np.log(x_span), label="log(x)")
#plt.ylim(-3, 3)

plt.legend()
plt.show()


