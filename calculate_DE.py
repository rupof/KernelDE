from squlearn.encoding_circuit.encoding_circuit_derivatives import *


import numpy as np
from scipy.integrate import solve_ivp
from MMR_implementation import DifferentialEquationSolver



####################333
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


##################################
            


 

# Create an instance of the DifferentialEquation
def f_analytical(x):
    lamb  = 20
    k = 0.1
    return np.exp(-lamb*x*k)*np.cos(lamb*x)

x_span = np.linspace(0, 1, 100)

x_initial = 0
f_initial = 1
solver = DifferentialEquationSolver(
    x_initial=x_initial,
    f_initial=f_initial,
    x_span = x_span,
    kernel_method_info = {"encoding_circuit_label":"Separable_rx", 
                            "executor_type":"statevector",
                            "method":"pre-calculated",
                            "num_shots":None,   
                            "kernel": np.array([rbf_kernel_manual(x_span, x_span, sigma=0.2), analytical_derivative_rbf_kernel(x_span, x_span, sigma=0.2)])
                           },     
        )

solver = DifferentialEquationSolver(
    x_initial=x_initial,
    f_initial=f_initial,
    x_span = x_span,
    kernel_method_info = {"encoding_circuit_label":"Separable_rx", 
                          "encoding_circuit_parameters": {"num_qubits": 2, "num_layers": 1, "rotation_gate":"rx"},
                            "executor_type":"statevector",
                            "method":"PQK",
                            "num_shots":None,   
                            "kernel": "auto"
                           },     
        )
# Solve the differential equation
solution, optimal_alpha = solver.solve()


import matplotlib.pyplot as plt
plt.plot(x_span, solution)
plt.plot(x_span, f_analytical(x_span), "--")
plt.show()



