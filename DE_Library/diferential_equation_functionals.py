import numpy as np

def loss_simple_test_QNN(f_alpha_tensor):
    """
    df/dx + sin(x) = 0
    f(0) = 1
    solution: f(x) = cos(x) 
    """
    x, f, dfdx, dfdxdx = f_alpha_tensor
    return dfdx + np.sin(x)

def derivatives_loss_simple_test_QNN(f_alpha_tensor, x_span):
    """
    df/dx + sin(x) = 0
    f(0) = 1
    """
    return [-np.sin(x_span)]

def grad_loss_simple_test_QNN(f_array):
    """
    n = x_span.shape[0] number of points
    m = x_span.shape[1] number of dimensions (typically m=1)

    F[x, x_, x__] = F(x, x_, x__)

    grad_F = (F(x, x_, x__)d(df), F(x, x_, x__)d(dfdx), F(x, x_, x__)d(dfdxdx))

    grad_F = (0, -1, 0)
    """
    x, f, dfdx, dfdxdx = f_array    
    dFdf = 0
    dFdfdx = 1
    dFdfdxdx = 0
    return [dFdf, dFdfdx, dFdfdxdx]
    
    
def loss_paper_decay_QNN(f_alpha_tensor):
    """
    df/dx + lamb * f*(k + tan(lamb*x)) = 0
    f(0) = 1
    solution: f(x) = np.exp(-lamb * x * k) * np.cos(lamb * x) + cte, f(0) = 1
    """
    x, f, dfdx, dfdxdx = f_alpha_tensor
    lamb = 8
    k = 0.1

    return dfdx + lamb * f * (k + np.tan(lamb*x))

def derivatives_loss_paper_decay_QNN(f_alpha_tensor, x):
        """
        df/dx + lamb * f*(k + tan(lamb*x)) = 0

        solution: f(x) = np.exp(-lamb * x * k) * np.cos(lamb * x), f(0) = 1
        """
        
        f = f_alpha_tensor[0]

        lamb = 8
        k = 0.1

        return [-lamb * f * (k + np.tan(lamb*x))]

def grad_loss_paper_decay_QNN(f_alpha_tensor):
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
    x, f, dfdx, dfdxdx = f_alpha_tensor

    dFdf = lamb * (k + np.tan(lamb*x))

    return [dFdf, 1, 0]



def derivatives_loss_paper(f_alpha_tensor, x):
        """
        0 = -lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) - lamb * k * f - df/dx

        solution: f(x) = np.exp(-lamb * x * k) * np.cos(lamb * x), f(0) = 1
        """
        f = f_alpha_tensor[0]

        lamb = 20
        k = 0.1

        return [-lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) - lamb * k * f]

def loss_paper(f_alpha_tensor):
    """
    0 = -lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) - lamb * k * f - df/dx
    solution: f(x) = np.exp(-lamb * x * k) * np.cos(lamb * x), f(0) = 1
    """
    x, f, dfdx, dfdxdx = f_alpha_tensor
    lamb = 20
    k = 0.1

    return lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) + lamb * k * f + dfdx

def grad_loss_paper(f_alpha_tensor):
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
    x, f, dfdx, dfdxdx = f_alpha_tensor

    dFdf = lamb*k 
    return [dFdf, 1, 0]
    

def loss_log_ode(f_alpha_tensor):
    """
    0 = -lamb * np.exp(f * k) + df/dx
    f(0.001) = np.log(0.001)

    solution: f(x) = np.log(x)
    """
    x, f, dfdx, dfdxdx = f_alpha_tensor 

    lamb = 1
    k = 1
    return dfdx - np.exp(-f*k)*lamb 

def grad_loss_log_ode(f_alpha_tensor):
    """
    n = x_span.shape[0] number of points
    m = x_span.shape[1] number of dimensions (typically m=1)

    F[x, x_, x__] = F(x, x_, x__)

    grad_F = (F(x, x_, x__)dx, F(x, x_, x__)dx_, F(x, x_, x__)dx__)

    F = -lamb * np.exp(-f * k) + df/dx

    grad_F = (lamb*k*np.exp(-f * k), 1, 0)
    """
    lamb = 1
    k = 1
    x, f, dfdx, dfdxdx = f_alpha_tensor

    dFdf = np.exp(-f*k)*lamb*k
    return [dFdf, 1, 0]

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


def loss_polynomial_with_exp(f_alpha_tensor):
    """
    0 = 2*f+4*cos(x)-8*sin(x) - df/dx 
    f(0) = 3

    solution: f(x) = 3*exp(2*x) + 4*sin(x)
    """
    x, f, dfdx, dfdxdx = f_alpha_tensor

    return 2*f+4*np.cos(x)-8*np.sin(x) - dfdx


def derivatives_loss_polynomial_with_exp(f_alpha_tensor, x):
    """
    0 = 2*f+4*cos(x)-8*sin(x) - df/dx 
    f(0) = 3

    solution: f(x) = 3*exp(2*x) + 4*sin(x)
    """
    f = f_alpha_tensor[0]

    return [2*f+4*np.cos(x)-8*np.sin(x)]

def loss_harmonic_oscillator(f_alpha_tensor):
    """
    L_functional = dfdx2 + f
    """
    x, f, dfdx, dfdxdx = f_alpha_tensor

    return dfdxdx + f

def grad_loss_harmonic_oscillator(f_alpha_tensor):
    """
    n = x_span.shape[0] number of points
    m = x_span.shape[1] number of dimensions (typically m=1)

    F[x, x_, x__] = F(x, x_, x__)

    grad_envelope = (F(x, x_, x__)dx, F(x, x_, x__)dx_, F(x, x_, x__)dx__)

    """
    x, f, dfdx, dfdxdx = f_alpha_tensor
    return [1, 0, 1]
    

def derivatives_loss_harmonic_oscillator(f_alpha_tensor, x_span):
    """
    L_functional = dfdx - g(f(x), x)
    """
    f = f_alpha_tensor[0]
    dfdx = f_alpha_tensor[1]
    return [dfdx, -f]


# Damped Harmonic Oscillator


def loss_damped_harmonic_oscillator(f_alpha_tensor):
    """
    L_functional = m*dfdxdx + c*dfdx + k*f
    """
    x, f, dfdx, dfdxdx = f_alpha_tensor
    k = 21**2
    c = 4
    return k*f + c*dfdx + dfdxdx

def grad_loss_damped_harmonic_oscillator(f_alpha_tensor):
    """
    n = x_span.shape[0] number of points
    m = x_span.shape[1] number of dimensions (typically m=1)

    F[x, x_, x__] = F(x, x_, x__)

    grad_F = (F(x, x_, x__)dx, F(x, x_, x__)dx_, F(x, x_, x__)dx__)

    grad_F = (k, c, 0)
    """
    k = 21**2
    c = 4
    x, f, dfdx, dfdxdx = f_alpha_tensor
    return [k, c, 0]

def derivatives_loss_damped_harmonic_oscillator(f_alpha_tensor, x_span):
    """
    L_functional = m*dfdxdx + c*dfdx + k*f
    """
    f = f_alpha_tensor[0]
    dfdx = f_alpha_tensor[1]
    k = 21**2
    c = 4

    return [dfdx, -c*dfdx - k*f]


mapping_of_loss_functions = {
    "paper": loss_paper,
    "log_ode": loss_log_ode,
    "polynomial_with_exp": loss_polynomial_with_exp,
    "harmonic_oscillator": loss_harmonic_oscillator,
    "paper_decay_QNN": loss_paper_decay_QNN, 
    "simple_test_QNN": loss_simple_test_QNN, 
    "damped_harmonic_oscillator": loss_damped_harmonic_oscillator
}

mapping_of_derivatives_of_loss_functions = {
    "paper": derivatives_loss_paper,
    "log_ode": derivatives_loss_log_ode,
    "polynomial_with_exp": derivatives_loss_polynomial_with_exp,
    "harmonic_oscillator": derivatives_loss_harmonic_oscillator,
    "paper_decay_QNN": derivatives_loss_paper_decay_QNN,
    "simple_test_QNN": derivatives_loss_simple_test_QNN, 
    "damped_harmonic_oscillator": derivatives_loss_damped_harmonic_oscillator
}

mapping_of_grad_of_loss_functions = {
    "paper": grad_loss_paper,
    "log_ode": grad_loss_log_ode,
    "polynomial_with_exp": derivatives_loss_polynomial_with_exp,
    "harmonic_oscillator": grad_loss_harmonic_oscillator,
    "paper_decay_QNN": grad_loss_paper_decay_QNN, 
    "simple_test_QNN": grad_loss_simple_test_QNN, 
    "damped_harmonic_oscillator": grad_loss_damped_harmonic_oscillator
}