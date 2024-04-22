import numpy as np



def rbf_fun(x,y,sigma=1):
    return np.exp(-(x-y)**2/(2*sigma**2))

def rbf_kernel_manual(x, y, sigma=1):
    kernel = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            kernel[i, j] = rbf_fun(x[i], y[j], sigma)
    return kernel

def analytical_derivative_rbf_kernel(x, y, sigma=1):
    """
    rbf_fun(x,y,sigma) = exp(-(x-y)**2/(2*sigma**2))
    df/dx = -exp(-(x-y)**2/(2*sigma**2)) * (2*(x-y)/(2*sigma**2))
    """
    derivative = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            derivative[i, j] = rbf_fun(x[i], y[j], sigma) * (-2*(x[i]-y[j])/(2*sigma**2))
    return derivative

def analytical_derivative_rbf_kernel_2(x, y, sigma=1):
    """
    rbf_fun(x,y,sigma) = exp(-(x-y)**2/(2*sigma**2))
    df/dx = -exp(-(x-y)**2/(2*sigma**2)) * (2*(x-y)/(2*sigma**2))
    d^2f/dx^2 = exp(-(x-y)**2/(2*sigma**2)) * (2*(x-y)/(2*sigma**2))**2 - exp(-(x-y)**2/(2*sigma**2)) * (2/(2*sigma**2))
    """
    derivative = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            #derivative[i, j] = rbf_dx[i,j] * -2*(x[i]-y[j])/(2*sigma**2) - rbf_fun(x[i], y[j], sigma) * (2/(2*sigma**2))
            derivative[i, j] = rbf_fun(x[i], y[j], sigma) * (2*(x[i]-y[j])/(2*sigma**2))**2 + rbf_fun(x[i], y[j], sigma) * (-2/(2*sigma**2))
    return derivative


