import numpy as np


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
        print("Kernel", len(f_alpha_tensor))
        x = x_span
        f = f_alpha_tensor[0]
        dfdx = f_alpha_tensor[1]
        try: 
            dfdxdx = f_alpha_tensor[2]
        except:
            dfdxdx = np.zeros_like(f)
        return x, f, dfdx, dfdxdx

    
def loss_simple_test_QNN(f_alpha_tensor, x_span = None):
    """
    df/dx + sin(x) = 0
    f(0) = 1
    solution: f(x) = cos(x) 
    """
    x, f, dfdx, dfdxdx = get_differentials(f_alpha_tensor, x_span)
    return dfdx + np.sin(x)

def derivatives_loss_simple_test_QNN(f_alpha_tensor, x_span = None):
        """
        df/dx + f = 0
        f(0) = 1
        """
        if len(f_alpha_tensor) == 1:
            f = f_alpha_tensor[0]
            x = x_span
        else:
            x, f, dfdx, dfdxdx = get_differentials(f_alpha_tensor, x_span)

        return [-np.sin(x)]
def grad_loss_simple_test_QNN(loss_values, x_span = None):
    """
    n = x_span.shape[0] number of points
    m = x_span.shape[1] number of dimensions (typically m=1)

    F[x, x_, x__] = F(x, x_, x__)

    grad_F = (F(x, x_, x__)dx, F(x, x_, x__)dx_, F(x, x_, x__)dx__)

    grad_F = (0, -1, 0)
    """
    x, f, dfdx, dfdxdx = get_differentials(loss_values, x_span)

    
    dfdp = loss_values["dfdp"] # shape (n, p)
    n_param = dfdp.shape[1]

    grad_envelope_list = np.zeros((3, x.shape[0], n_param)) # shape (3, n, p)
    grad_envelope_list[0,:,:] = 0 # (dF/df,... p times) 
    grad_envelope_list[1,:,:] = 1  # dF/dfdx
    grad_envelope_list[2,:,:] =  0  # dF/dfdxdx
    return grad_envelope_list
    
def loss_paper_decay_QNN(f_alpha_tensor, x_span = None):
    """
    df/dx + lamb * f*(k + tan(lamb*x)) = 0
    f(0) = 1
    solution: f(x) = np.exp(-lamb * x * k) * np.cos(lamb * x) + cte, f(0) = 1
    """
    x, f, dfdx, dfdxdx = get_differentials(f_alpha_tensor, x_span)
    lamb = 8
    k = 0.1

    return dfdx + lamb * f * (k + np.tan(lamb*x))

def derivatives_loss_paper_decay_QNN(f_alpha_tensor, x_span = None):
        """
        df/dx + lamb * f*(k + tan(lamb*x)) = 0

        solution: f(x) = np.exp(-lamb * x * k) * np.cos(lamb * x), f(0) = 1
        """
        if len(f_alpha_tensor) == 1:
            f = f_alpha_tensor[0]
            x = x_span
        else:
            x, f, dfdx, dfdxdx = get_differentials(f_alpha_tensor, x_span)

        lamb = 8
        k = 0.1

        return [-lamb * f * (k + np.tan(lamb*x))]

def grad_loss_paper_decay_QNN(loss_values, x_span = None):
    """
    n = x_span.shape[0] number of points
    m = x_span.shape[1] number of dimensions (typically m=1)

    F[x, x_, x__] = F(x, x_, x__)

    grad_F = (F(x, x_, x__)dx, F(x, x_, x__)dx_, F(x, x_, x__)dx__)

    F = dfdx + lamb * f * (k + np.tan(lamb*x))

    grad_F = (-lamb*k, -1, 0)
    """
    lamb = 8
    k = 0.1
    x, f, dfdx, dfdxdx = get_differentials(loss_values, x_span)

    
    dfdp = loss_values["dfdp"] # shape (n, p)
    n_param = dfdp.shape[1]

    dFdf = lamb * (k + np.tan(lamb*x))

    grad_envelope_list = np.zeros((3, x.shape[0], n_param)) # shape (3, n, p)
    grad_envelope_list[0,:,:] = np.tile(dFdf, (n_param, 1)).T  
    grad_envelope_list[1,:,:] = 1  # dF/dfdx
    grad_envelope_list[2,:,:] =  0  # dF/dfdxdx
    return grad_envelope_list

def derivatives_loss_paper(f_alpha_tensor, x_span = None):
        """
        0 = -lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) - lamb * k * f - df/dx

        solution: f(x) = np.exp(-lamb * x * k) * np.cos(lamb * x), f(0) = 1
        """
        if len(f_alpha_tensor) == 1:
            f = f_alpha_tensor[0]
            x = x_span
        else:
            x, f, dfdx, dfdxdx = get_differentials(f_alpha_tensor, x_span)

        lamb = 20
        k = 0.1

        return [-lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) - lamb * k * f]

def loss_paper(f_alpha_tensor, x_span = None):
    """
    0 = -lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) - lamb * k * f - df/dx
    solution: f(x) = np.exp(-lamb * x * k) * np.cos(lamb * x), f(0) = 1
    """
    x, f, dfdx, dfdxdx = get_differentials(f_alpha_tensor, x_span)
    lamb = 20
    k = 0.1

    return lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) + lamb * k * f + dfdx

def grad_loss_paper(loss_values, x_span = None):
    """
    n = x_span.shape[0] number of points
    m = x_span.shape[1] number of dimensions (typically m=1)

    F[x, x_, x__] = F(x, x_, x__)

    grad_F = (F(x, x_, x__)dx, F(x, x_, x__)dx_, F(x, x_, x__)dx__)

    -lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) - lamb * k * f - dfdx

    grad_F = (-lamb*k, -1, 0)
    """
    lamb = 20
    k = 0.1
    x, f, dfdx, dfdxdx = get_differentials(loss_values, x_span)

    
    dfdp = loss_values["dfdp"] # shape (n, p)
    n_param = dfdp.shape[1]

    grad_envelope_list = np.zeros((3, x.shape[0], n_param)) # shape (3, n, p) 
    grad_envelope_list[0,:,:] =  lamb*k # (dF/df,... p times) 
    grad_envelope_list[1,:,:] =  1  # dF/dfdx
    grad_envelope_list[2,:,:] =  0  # dF/dfdxdx
    return grad_envelope_list

def loss_log_ode(f_alpha_tensor, x_span = None):
    """
    0 = lamb * np.exp(f * k) - df/dx
    f(0.001) = np.log(0.001)

    solution: f(x) = np.log(x)
    """
    if x_span is None: #Then, we are using a QNN
        loss_values = f_alpha_tensor #f_alpha_tensor is the loss_values dictionary
        x = loss_values["x"]
        f = loss_values["f"]
        dfdx = loss_values["dfdx"][:,0]
    else: #Then, we are using a Kernel
        x = x_span
        f = f_alpha_tensor[0]
        dfdx = f_alpha_tensor[1]

    lamb = 1
    k = 1
    return dfdx - np.exp(-f*k)*lamb 

def grad_loss_log_ode(loss_values, x_span = None):
    """
    n = x_span.shape[0] number of points
    m = x_span.shape[1] number of dimensions (typically m=1)

    F[x, x_, x__] = F(x, x_, x__)

    grad_F = (F(x, x_, x__)dx, F(x, x_, x__)dx_, F(x, x_, x__)dx__)

    F = lamb * np.exp(f * k) - df/dx

    grad_F = (-lamb*k, -1, 0)
    """
    lamb = 1
    k = 1
    x, f, dfdx, dfdxdx = get_differentials(loss_values, x_span)

    
    dfdp = loss_values["dfdp"] # shape (n, p)
    n_param = dfdp.shape[1]

    dFdf = lamb*k*np.exp(-f*k)

    grad_envelope_list = np.zeros((3, x.shape[0], n_param)) # shape (3, n, p)
    grad_envelope_list[0,:,:] = np.tile(dFdf, (n_param, 1)).T  
    grad_envelope_list[1,:,:] =  1  # dF/dfdx
    grad_envelope_list[2,:,:] =  0  # dF/dfdxdx
    return grad_envelope_list

def derivatives_loss_log_ode(f_alpha_tensor, x_span = None):
    """
    0 = lamb * np.exp(f * k) - df/dx
    f(0.001) = np.log(0.001)

    solution: f(x) = np.log(x)
    """
    f = f_alpha_tensor[0]
    lamb = 1
    k = 1
    return [np.exp(-f*k)*lamb]


def loss_polynomial_with_exp(f_alpha_tensor, x_span = None):
    """
    0 = 2*f+4*cos(x)-8*sin(x) - df/dx 
    f(0) = 3

    solution: f(x) = 3*exp(2*x) + 4*sin(x)
    """
    x, f, dfdx, dfdxdx = get_differentials(f_alpha_tensor, x_span)

    return 2*f+4*np.cos(x)-8*np.sin(x) - dfdx


def derivatives_loss_polynomial_with_exp(f_alpha_tensor, x_span = None):
    """
    0 = 2*f+4*cos(x)-8*sin(x) - df/dx 
    f(0) = 3

    solution: f(x) = 3*exp(2*x) + 4*sin(x)
    """
    f = f_alpha_tensor[0]
    x = x_span

    return [2*f+4*np.cos(x)-8*np.sin(x)]

def loss_harmonic_oscillator(f_alpha_tensor, x_span = None):
    """
    L_functional = dfdx2 + f
    """
    x, f, dfdx, dfdxdx = get_differentials(f_alpha_tensor, x_span)

    return dfdxdx + f

def grad_loss_harmonic_oscillator(f_alpha_tensor, x_span = None):
    """
    n = x_span.shape[0] number of points
    m = x_span.shape[1] number of dimensions (typically m=1)

    F[x, x_, x__] = F(x, x_, x__)

    grad_envelope = (F(x, x_, x__)dx, F(x, x_, x__)dx_, F(x, x_, x__)dx__)

    """
    loss_values = f_alpha_tensor
    x, f, dfdx, dfdxdx = get_differentials(f_alpha_tensor, x_span)

    #x_span = loss_values["x"] # shape (n, m)   
    #f = loss_values["f"] # shape (n,)
    dfdp = loss_values["dfdp"] # shape (n, p)
    n_param = dfdp.shape[1]

    grad_envelope_list = np.zeros((3, x.shape[0], n_param)) # shape (3, n, p) 
    grad_envelope_list[0,:,:] = 1 # (dF/df,... p times) 
    grad_envelope_list[1,:,:] = 0  # dF/dfdx
    grad_envelope_list[2,:,:] = 1  # dF/dfdxdx

    #f = loss_values["f"] # shape (n,)
    #dfdx = loss_values["dfdx"][:,0] # shape (n, m) 
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


def derivatives_loss_harmonic_oscillator(f_alpha_tensor, x_span):
    """
    L_functional = dfdx - g(f(x), x)
    """
    f = f_alpha_tensor[0]
    dfdx = f_alpha_tensor[1]
    return [dfdx, -f]

mapping_of_loss_functions = {
    "paper": loss_paper,
    "log_ode": loss_log_ode,
    "polynomial_with_exp": loss_polynomial_with_exp,
    "harmonic_oscillator": loss_harmonic_oscillator,
    "paper_decay_QNN": loss_paper_decay_QNN, 
    "simple_test_QNN": loss_simple_test_QNN
}

mapping_of_derivatives_of_loss_functions = {
    "paper": derivatives_loss_paper,
    "log_ode": derivatives_loss_log_ode,
    "polynomial_with_exp": derivatives_loss_polynomial_with_exp,
    "harmonic_oscillator": derivatives_loss_harmonic_oscillator,
    "paper_decay_QNN": derivatives_loss_paper_decay_QNN,
    "simple_test_QNN": derivatives_loss_simple_test_QNN
}

mapping_of_grad_of_loss_functions = {
    "paper": grad_loss_paper,
    "log_ode": grad_loss_log_ode,
    "polynomial_with_exp": derivatives_loss_polynomial_with_exp,
    "harmonic_oscillator": grad_loss_harmonic_oscillator,
    "paper_decay_QNN": grad_loss_paper_decay_QNN, 
    "simple_test_QNN": grad_loss_simple_test_QNN
}