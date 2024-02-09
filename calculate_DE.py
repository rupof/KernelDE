from squlearn.encoding_circuit.encoding_circuit_derivatives import *


import numpy as np
from scipy.integrate import odeint
from solvers.MMR.kernel_solver import Solver
from solvers.MMR.PQK_solver import PQK_solver
from solvers.MMR.FQK_solver import FQK_solver

from utils.rbf_kernel_tools import *
from circuits.circuits import *

import matplotlib.pyplot as plt



def g(f, x):
        lamb = 20
        k = 0.1
        return -lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) - lamb * k * f


####################################3

x_span = np.linspace(0, 1, 21)
f_initial = 1

######################3


#Numerical and Analytical Solutions
f_odeint = odeint(g, f_initial, x_span[:])
#f_analytical_sol = f_analytical_fun(x_span)

#Classical Solver
#RBF
RBF_kernel_list = [rbf_kernel_manual(x_span, x_span, sigma = 0.2), analytical_derivative_rbf_kernel(x_span, x_span, sigma = 0.2)]
Solver_test = Solver(RBF_kernel_list)
f_RBF, optimal_alpha_RBF = Solver_test.solve(x_span, f_initial, g)

#PQK
PQK_solver_test = PQK_solver({"encoding_circuit": HardwareEfficientEmbeddingCircuit_qiskit, 
                              "num_qubits": 8,
                              "num_layers": 2,
                              "rotation_gate":"rx",},
                              Executor("statevector_simulator"), 
                              envelope={"function": rbf_kernel_manual, 
                                        "derivative_function": analytical_derivative_rbf_kernel, 
                                        "sigma": 1})

solution_PQK, kernel_list_PQK = PQK_solver_test.solver(x_span, f_initial, g)
f_PQK = solution_PQK[0]
optimal_alpha_PQK = solution_PQK[1]


FQK_solver_test = FQK_solver({"encoding_circuit": HardwareEfficientEmbeddingCircuit_qiskit, 
                              "num_qubits": 6,
                              "num_layers": 2,
                              "rotation_gate":"rx",},
                              Executor("statevector_simulator"))
solution_FQK, kernel_listFQK = FQK_solver_test.solver(x_span, f_initial, g)
f_FQK = solution_FQK[0]
optimal_alpha_FQK = solution_FQK[1]





x_span_plot = x_span.reshape(-1, 1)
plt.plot(x_span_plot, f_odeint, label="odeint")
plt.plot(x_span, f_RBF, label="RBF")
plt.plot(x_span, f_PQK, label="PQK")
plt.plot(x_span_plot, f_FQK, label="FQK")

plt.legend()
plt.show()


