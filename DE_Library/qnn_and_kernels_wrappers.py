import numpy as np
from squlearn.qnn import ODELoss
from squlearn import Executor




def get_differentials(f_alpha_tensor, x_span = None):

    if x_span is None: #Then, we are using a QNN
        loss_values = f_alpha_tensor #f_alpha_tensor is the loss_values dictionary
        x = loss_values["x"]
        f = loss_values["f"]
        dfdx = loss_values["dfdx"][:,0]
        try: 
            dfdxdx = loss_values["dfdxdx"][:,0,0]
            return x, f, dfdx, dfdxdx
        except:
            return x, f, dfdx, np.zeros_like(f)
    else: #Then, we are using a Kernel
        x = x_span
        f = f_alpha_tensor[0]
        dfdx = f_alpha_tensor[1]
        try: 
            dfdxdx = f_alpha_tensor[2]
        except:
            dfdxdx = np.zeros_like(f)
        return x, f, dfdx, dfdxdx

def create_QNN_loss(loss_functional):
    def QNN_loss(QNN_derivatives_values):
        """
        Defines the loss function for the ODE problem
        f_array is assumed to be [x, f, dfdx, dfdxdx]
        
        """
        return loss_functional(get_differentials(QNN_derivatives_values))
    return QNN_loss

def create_QNN_gradient(ODE_functional_gradient):
    def QNN_gradient(QNN_derivatives_values):
        """
        Defines the gradient of the loss function for the ODE problem
        f_array is assumed to be [x, f, dfdx, dfdxdx]
        
        """
        dFdf, dFdfdx, dFdfdxdx = ODE_functional_gradient(get_differentials(QNN_derivatives_values))

        dfdp = QNN_derivatives_values["dfdp"] # shape (n, p)

        x = QNN_derivatives_values["x"] # shape (n, m)
        n_param = dfdp.shape[1]
        
        grad_envelope_list = np.zeros((3, x.shape[0], n_param)) # shape (3, n, p)
        grad_envelope_list[0,:,:] = np.tile(dFdf, (n_param, 1)).T  
        grad_envelope_list[1,:,:] = np.tile(dFdfdx, (n_param, 1)).T
        grad_envelope_list[2,:,:] =  np.tile(dFdfdxdx, (n_param, 1)).T
        return grad_envelope_list
    return QNN_gradient

def ODELoss_wrapper(ODE_functional, ODE_functional_gradient, initial_vec, eta = 1.0, boundary_handling = "pinned", true_solution = None):
    return ODELoss(create_QNN_loss(ODE_functional), create_QNN_gradient(ODE_functional_gradient), initial_vec, eta, boundary_handling, true_solution)
    

executor_type_dictionary = {
    "statevector_simulator": Executor("statevector_simulator"),
    "pennylane": Executor("pennylane"), 
    "qasm_simulator_variance": Executor("qasm_simulator", shots=5000, seed=1),
    "pennylane_shots_variance": Executor("default.qubit", shots=5000, seed = 1),
    "qasm_simulator": Executor("qasm_simulator", shots=5000, seed=1),
    "pennylane_shots": Executor("default.qubit", shots=5000, seed = 1),
    "qiskit_shots_variance": Executor("qiskit", shots=5000, seed=1),
    "qiskit_shots": Executor("qiskit", shots=5000, seed=1)
}