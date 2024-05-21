
import numpy as np
from squlearn import Executor
from squlearn.encoding_circuit import *
from squlearn.observables import SummedPaulis
from squlearn.qnn import QNNRegressor, ODELoss
from squlearn.qnn.lowlevel_qnn import LowLevelQNN
from squlearn.optimizers import SLSQP, Adam
from squlearn.qnn.loss import *
from squlearn.qnn.training import *

# General form of the differential equation is given by:
# f'(x) = g(f, x)



def g_exp(f, x):
    """
    df/dx = lamb * np.exp(-f * k) 
    f(0.001) = np.log(0.001)

    solution: f(x) = np.log(x)

    #dg/df = - lamb * k * np.exp(-f * k)
    """
    lamb = 1
    k = 1
    return np.exp(-f*k)*lamb

####################################3


######################3
def L_functional(loss_values):
    x = loss_values["x"]
    f = loss_values["f"]
    dfdx = loss_values["dfdx"][:,0]
    #dfdxdx = loss_values["dfdxdx"][:,0,0]
    value =  dfdx - g_exp(f,x)    #d_ODE_functional_dD = (0, 1, 0)
    #insert initial value at last position of value
    #value = np.append(value, initial_value)
    return value

def grad_F_functional(loss_values):
    """ArithmeticError
    n = x_span.shape[0] number of points
    m = x_span.shape[1] number of dimensions (typically m=1)

    F[x, x_, x__] = envelope(x, x_, x__)

    grad_envelope = (envelope(x, x_, x__)dx, envelope(x, x_, x__)dx_, envelope(x, x_, x__)dx__)

    """
    x_span = loss_values["x"] # shape (n, m)   
    f = loss_values["f"] # shape (n,)
    dfdp = loss_values["dfdp"] # shape (n, p)
    n_param = dfdp.shape[1]

    dFdf = -1*np.exp(-f[:])  # dF/df
    grad_envelope_list = np.zeros((3, x_span.shape[0], n_param)) # shape (3, n, p) 
    grad_envelope_list[0,:,:] = np.tile(dFdf, (n_param, 1)).T # (dF/df,... p times) 
    grad_envelope_list[1,:,:] = 1  # dF/dfdx
    grad_envelope_list[2,:,:] = 0  # dF/dfdxdx

    f = loss_values["f"] # shape (n,)
    dfdx = loss_values["dfdx"][:,0] # shape (n, m) 
    #dfdxdx = loss_values["dfdxdx"][:,0,:] # shape (n, 1, m)
    #dfdpdx = loss_values["dfdpdx"][:,0,:] # shape (n, 1, p)
    #dfdpdxdx = loss_values["dfdpdxdx"][:,0,:] # shape (n, 1, 1, p)
    
    try:
        dfdxdp = loss_values["dfdxdp"] # shape (n, 1, P)
        #dfdxdxdp = loss_values["dfdxdxdp"] # shape (n, 1, 1, P)
    except:
        dfdpdx = loss_values["dfdpdx"][:, :, 0] # shape (n, p, 1)
        #dfdpdxdx = loss_values["dfdpdxdx"][:,:, 0, 0] # shape (n, p, 1, 1)

    #problem dependent gradient!!
    
    ###########3

    return grad_envelope_list


#############3

x_line = np.linspace(0.1, 1.5, 20)
num_qubits = 3
num_features = 1
num_layers = 3
initial_value = [np.log(x_line[0])]
np.random.seed(0)
param_ini = np.random.rand(2*num_qubits*num_layers)



loss_ODE = ODELoss(L_functional, grad_F_functional, initial_vec = initial_value, eta=1)
loss = SquaredLoss()
slsqp = SLSQP(options={"maxiter": 150, "ftol": 0.009})
adam = Adam(options={"maxiter": 15, "tol": 0.00009})


clf = QNNRegressor(
    ChebyshevTower(num_qubits, num_features, num_layers= num_layers),
    SummedPaulis(num_qubits),
    Executor("pennylane"),
    loss_ODE,
    adam,
    param_ini,
    np.ones(num_qubits+1),
    opt_param_op = False
)    

y_ODE = np.zeros((x_line.shape[0]))
clf._fit(x_line, y_ODE,  weights=None)
y_pred = clf.predict(x_line)


import sys, os 

results_path = "./data/results/" #./data/results/ for local, /data/results/ for server
index = str(sys.argv[1])


results_folder_path = results_path + f"QNN_tests_DE_{index}"
if not os.path.exists(results_folder_path):
    os.makedirs(results_folder_path)

dict_to_save = {"x": x_line, "y": y_pred, "params": clf.param}

np.save(results_folder_path + f"/QNN_DE.npy", dict_to_save)