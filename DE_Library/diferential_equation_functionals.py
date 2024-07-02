import numpy as np

def loss_to_ivp_loss(derivative_loss):
    def ivp_loss(x, f):
        dfdx = derivative_loss([f], x)
        return dfdx
    return ivp_loss


def loss_simple_test_QNN(f_alpha_tensor):
    """
    df/dx + sin(x) = 0
    f(0) = 1
    solution: f(x) = cos(x) 
    """
    x, f, dfdx, dfdxdx = f_alpha_tensor
    return dfdx + np.sin(12*x)

def derivatives_loss_simple_test_QNN(f_alpha_tensor, x_span):
    """
    df/dx + sin(x) = 0
    f(0) = 1
    """
    return [-np.sin(12*x_span)]

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
    dFdfdx = 1
    dFdfdxdx = 0


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

def grad_loss_polynomial_with_exp(f_alpha_tensor):
    """
    n = x_span.shape[0] number of points
    m = x_span.shape[1] number of dimensions (typically m=1)

    F[x, x_, x__] = F(x, x_, x__)

    grad_F = (F(x, x_, x__)dx, F(x, x_, x__)dx_, F(x, x_, x__)dx__)

    grad_F = (0, -1, 0)
    """
    x, f, dfdx, dfdxdx = f_alpha_tensor
    return [2, -1, 0]

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

def loss_paper_nontrivial_dynamics(f_alpha_tensor):
    """
    f_0 = 0.75
    """
    x, f, dfdx, dfdxdx = f_alpha_tensor

    return dfdx -4*f + 6*f**2 - np.sin(50*x)-f*np.cos(25*x) + 0.5 

def grad_paper_nontrivial_dynamics(f_alpha_tensor):
    """
    grad_F = (-4 - np.cos(25*x), 1, 0)
    """
    x, f, dfdx, dfdxdx = f_alpha_tensor
    return [-4 - np.cos(25*x) + 12*f, 1, 0]

def derivatives_loss_paper_nontrivial_dynamics(f_alpha_tensor, x_span):
    """
    f_0 = 0.75
    """
    f = f_alpha_tensor[0]

    return [4*f - 6*f**2 + np.sin(50*x_span)+f*np.cos(25*x_span) - 0.5]


def loss_error_function_ode(f_alpha_tensor):
    """
    Zill p.60
    y(0) = 1

    Analytical solution:
    y(x) = e^(-x^2)\left(1 + 2 \int_0^x e^{t^2} dt\right)
    """
    x, f, dfdx, dfdxdx = f_alpha_tensor

    return dfdx - 2*x*f 

def grad_error_function_ode(f_alpha_tensor):
    """
    grad_F = (F(x, x_, x__)dx, F(x, x_, x__)dx_, F(x, x_, x__)dx__)

    grad_F = (-2*x, 0, 1)
    """
    x, f, dfdx, dfdxdx = f_alpha_tensor
    return [-2*x, 1, 0]

def derivatives_loss_error_function_ode(f_alpha_tensor, x_span):    
    """
    y(0) = 1

    Analytical solution:
    y(x) = e^(-x^2)\left(1 + 2 \int_0^x e^{t^2} dt\right)
    """
    f = f_alpha_tensor[0]

    return [2*x_span*f]

def loss_bernoulli_DE(f_alpha_tensor):
    """
    Example 3, p. 74
    dfdx - (-2*x + f)**2 - 7
    dfdx - f**2 + 4*f*x - 4*x**2 - 7
    y(0) = 0
    Analytical solution:
    y = 2*x + 3*(1-e^6x)/(1 + e^6x)
    """
    x, f, dfdx, dfdxdx = f_alpha_tensor
    return dfdx - (-2*x + f)**2 - 7

def grad_bernoulli_DE(f_alpha_tensor):
    """
    grad_F = (F(x, x_, x__)dx, F(x, x_, x__)dx_, F(x, x_, x__)dx__)

    grad_F = (-2*(2*x + 3*(1-np.exp(6*x))/(1 + np.exp(6*x))), 1, 0)
    """
    x, f, dfdx, dfdxdx = f_alpha_tensor
    return [2*f + 4, 1, 0]

def derivatives_loss_bernoulli_DE(f_alpha_tensor, x_span):
    """
    Example 3, p. 74
    dfdx - (-2*x + f)**2 - 7
    dfdx - f**2 + 4*f*x - 4*x**2 - 7
    y(0) = 0
    Analytical solution:
    y = 2*x + 3*(1-e^6x)/(1 + e^6x)
    """
    f = f_alpha_tensor[0]
    return [(-2*x_span + f)**2 + 7]


def loss_arbitrary_ode(f_alpha_tensor):
    """
    initial condition: f(0) = 0
    """
    x, f, dfdx, dfdxdx = f_alpha_tensor
    return -f + x**3 + x**2 + np.sin(f*30) -dfdx

def grad_arbitrary_ode(f_alpha_tensor):
    """
    n = x_span.shape[0] number of points
    m = x_span.shape[1] number of dimensions (typically m=1)

    F[x, x_, x__] = F(x, x_, x__)

    grad_F = (F(x, x_, x__)dx, F(x, x_, x__)dx_, F(x, x_, x__)dx__)

    grad_F = (3*x**2 + 2*x + 30*np.cos(f*30), -1, 0)
    """
    x, f, dfdx, dfdxdx = f_alpha_tensor
    return [-1 +  30*np.cos(f*30), -1, 0]

def derivatives_loss_arbitrary_ode(f_alpha_tensor, x_span):
    """
    L_functional = dfdx - g(f(x), x)
    """
    f = f_alpha_tensor[0]
    return [-f + x_span**3 + x_span**2 + np.sin(f*30)]



def loss_arbitrary_ode(x, f):
    dfdx = k/a*f*(a -f)
    return dfdx

# Initial condition
y0 = [0.1]

def loss_logistic_equation(f_alpha_tensor):
    """
    L_functional = k/a*f*(a - f) = k*f - k*f**2/a
    """
    a, k = 10, 10
    x, f, dfdx, dfdxdx = f_alpha_tensor
    return k/a*f*(a - f)-dfdx

def grad_logistic_equation(f_alpha_tensor):
    """
    grad_F = (F(x, x_, x__)dx, F(x, x_, x__)dx_, F(x, x_, x__)dx__)

    grad_F = (k - 2*k*f/a, k/a, 0)
    """
    a, k = 10, 10
    x, f, dfdx, dfdxdx = f_alpha_tensor
    return [k - 2*k*f/a, k/a, -1]

def derivatives_loss_logistic_equation(f_alpha_tensor, x_span):
    """
    L_functional = k/a*f*(a - f) = k*f - k*f**2/a
    """
    a, k = 10, 10
    f = f_alpha_tensor[0]
    return [k/a*f*(a - f)]


mapping_of_loss_functions = {
    "paper": loss_paper,
    "log_ode": loss_log_ode,
    "polynomial_with_exp": loss_polynomial_with_exp,
    "harmonic_oscillator": loss_harmonic_oscillator,
    "paper_decay_QNN": loss_paper_decay_QNN, 
    "simple_test_QNN": loss_simple_test_QNN, 
    "damped_harmonic_oscillator": loss_damped_harmonic_oscillator,
    "paper_nontrivial_dynamics": loss_paper_nontrivial_dynamics,
    "error_function_ode": loss_error_function_ode, 
    "bernoulli_DE": loss_bernoulli_DE, 
    "arbitrary_ode": loss_arbitrary_ode,
    "logistic_equation": loss_logistic_equation,

}

mapping_of_derivatives_of_loss_functions = {
    "paper": derivatives_loss_paper,
    "log_ode": derivatives_loss_log_ode,
    "polynomial_with_exp": derivatives_loss_polynomial_with_exp,
    "harmonic_oscillator": derivatives_loss_harmonic_oscillator,
    "paper_decay_QNN": derivatives_loss_paper_decay_QNN,
    "simple_test_QNN": derivatives_loss_simple_test_QNN, 
    "damped_harmonic_oscillator": derivatives_loss_damped_harmonic_oscillator,
    "paper_nontrivial_dynamics": derivatives_loss_paper_nontrivial_dynamics,
    "error_function_ode": derivatives_loss_error_function_ode,
    "bernoulli_DE": derivatives_loss_bernoulli_DE,
    "arbitrary_ode": derivatives_loss_arbitrary_ode,
    "logistic_equation": derivatives_loss_logistic_equation,
}

mapping_of_grad_of_loss_functions = {
    "paper": grad_loss_paper,
    "log_ode": grad_loss_log_ode,
    "polynomial_with_exp": grad_loss_polynomial_with_exp,
    "harmonic_oscillator": grad_loss_harmonic_oscillator,
    "paper_decay_QNN": grad_loss_paper_decay_QNN, 
    "simple_test_QNN": grad_loss_simple_test_QNN, 
    "paper_nontrivial_dynamics": grad_paper_nontrivial_dynamics,
    "damped_harmonic_oscillator": grad_loss_damped_harmonic_oscillator,
    "error_function_ode": grad_error_function_ode,
    "bernoulli_DE": grad_bernoulli_DE,
    "arbitrary_ode": grad_arbitrary_ode,
    "logistic_equation": grad_logistic_equation,
}


#1rst order ODEs list:
#1. Paper: df/dx + lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) + lamb * k * f = 0, f(0) = 1 paper
#2. Log ODE: df/dx - lamb * np.exp(f * k) = 0, f(0.01) = np.log(0.01) log_ode
#3. Polynomial with exp: 2*f+4*cos(x)-8*sin(x) - df/dx = 0, f(0) = 3 polynomial_with_exp
#4. Harmonic oscillator: dfdx2 + f = 0, f(0) = 1 harmonic_oscillator
#5. Paper decay QNN: df/dx + lamb * f*(k + tan(lamb*x)) = 0, f(0) = 1 paper_decay_QNN
#6. Simple Cosine QNN: df/dx + sin(x) = 0, f(0) = 1 simple_test_QNN
#7. Paper nontrivial dynamics: dfdx -4*f + 6*f**2 - np.sin(50*x)-f*np.cos(25*x) + 0.5 = 0, f(0) = 0.75 paper_nontrivial_dynamics
#8. Error function ODE: dfdx - 2*x*f = 0, f(0) = 0 error_function_ode
#9. Bernoulli DE: dfdx - (-2*x + f)**2 - 7 = 0, f(0) = 0 bernoulli_DE
#10. Arbitrary ODE: -f + x**3 + x**2 + np.sin(f*30) -dfdx = 0, f(0) = 0 arbitrary_ode



##### I will use: 
#1. Paper: df/dx + lamb * np.exp(-lamb * x * k) * np.sin(lamb * x) + lamb * k * f = 0, f(0) = 1 paper
#2. Log ODE: df/dx - lamb * np.exp(f * k) = 0, f(0.01) = np.log(0.01) log_ode
#3. Polynomial with exp: 2*f+4*cos(x)-8*sin(x) - df/dx = 0, f(0) = 3 polynomial_with_exp
#4. Paper decay QNN: df/dx + lamb * f*(k + tan(lamb*x)) = 0, f(0) = 1 paper_decay_QNN
#5. Simple test QNN: df/dx + sin(x) = 0, f(0) = 1 simple_test_QNN
#6. Paper nontrivial dynamics: dfdx -4*f + 6*f**2 - np.sin(50*x)-f*np.cos(25*x) + 0.5 = 0, f(0) = 0.75 paper_nontrivial_dynamics
#7. Bernoulli DE: dfdx - (-2*x + f)**2 - 7 = 0, f(0) = 0 bernoulli_DE
#8. Arbitrary ODE: -f + x**3 + x**2 + np.sin(f*30) -dfdx = 0, f(0) = 0  arbitrary_ode