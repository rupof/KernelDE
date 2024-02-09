from squlearn.encoding_circuit.encoding_circuit_derivatives import *


import numpy as np
from scipy.integrate import odeint
from solvers.MMR.kernel_solver import *
from solvers.MMR.PQK_solver import *

 
def g(f, x):
        lamb = 20
        k = 0.1
        return -lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) - lamb * k * f


def rbf_fun(x,y,sigma=1):
    return np.exp(-(x-y)**2/(2*sigma**2))

def rbf_kernel_manual(x, y, sigma=1):
    kernel = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            kernel[i, j] = rbf_fun(x[i], y[j], sigma)
    return kernel

def analytical_derivative_rbf_kernel(x, y, sigma=1):
    derivative = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            derivative[i, j] = -rbf_fun(x[i], y[j], sigma) * (2*(x[i]-y[j])/(2*sigma**2))
    return derivative




####################################3

x_span = np.linspace(0, 1, 20)
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



#plot all solutions


import matplotlib.pyplot as plt


x_span = x_span.reshape(-1, 1)
plt.plot(x_span, f_odeint, label="odeint")
plt.plot(x_span, f_RBF, label="RBF")
plt.plot(x_span, f_PQK, label="PQK")

plt.legend()
plt.show()


