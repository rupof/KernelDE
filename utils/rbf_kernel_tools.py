import numpy as np
 #from sklearn import RBF 

from sklearn.gaussian_process.kernels import RBF
from scipy.spatial.distance import cdist




def rbf_fun(x,y,sigma=1):
    return np.exp(-(x-y)**2/(2*sigma**2))

def rbf_kernel_manual(x, y, sigma=1):
    kernel = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            kernel[i, j] = rbf_fun(x[i], y[j], sigma)
    return kernel

def analytical_derivative_rbf_kernel(x, y, sigma=1):
    """-
    assumes x, y are 1D arrays
    rbf_fun(x,y,sigma) = exp(-(x-y)**2/(2*sigma**2))
    df/dx = -exp(-(x-y)**2/(2*sigma**2)) * (2*(x-y)/(2*sigma**2))
    """
    derivative = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            derivative[i, j] = rbf_fun(x[i], y[j], sigma) * (-2*(x[i]-y[j])/(2*sigma**2))
    return derivative

def matrix_rbf(X , Y, sigma = 1):
    return RBF(sigma)(X, Y)

def matrix_rbf_dx_slow(X, Y, sigma=1):
    """
    $K(\vec{x},\vec{y}) = exp(-gamma*||\vec{x}-\vec{y}||^2)$
    dK/dx_i = -2*gamma*(x_i-y_i)*K(\vec{x},\vec{y})
    """    
    gamma = 1/(2*sigma**2)
    n, d = X.shape
    
    gram_dx = np.zeros((n, n, d))
    #broadcasting
    for l in range(d):
        for i in range(n):
            for j in range(n):
                gram_dx[i, j, l] = -2*gamma*(X[i,l]-Y[j,l])*RBF(sigma)([X[i]], [Y[j]])[0,0]
    return gram_dx



def matrix_rbf_dxdx_slow(X, Y, sigma=1):
    """
    $K(\vec{x},\vec{y}) = exp(-gamma*||\vec{x}-\vec{y}||^2)$
    dK/dx_ldx_l = -2*gamma*(x_i-y_i)*K(\vec{x},\vec{y})
    """    
    gamma = 1/(2*sigma**2)
    n, d = X.shape
    
    gram_dx = np.zeros((n, n, d))
    #broadcasting
    for l in range(d):
        for i in range(n):
            for j in range(n):
                gram_dx[i, j, l] = 2*gamma*(2*gamma*(X[i,l]-Y[j,l])**2-1)*RBF(sigma)([X[i]], [Y[j]])[0,0]
    return gram_dx

def matrix_rbf_dxdy_slow(X, Y, sigma=1):
    """
    mixed derivative of the RBF kernel
    dK/dx_ldx_p
    """    
    gamma = 1/(2*sigma**2)
    n, d = X.shape
    
    gram_dx = np.zeros((n, n, d, d))
    print("HALL1OO")

    #broadcasting
    for l in range(d):
        for p in range(d):
            for i in range(n):
                for j in range(n):
                    gram_dx[i, j, l, p] = 4*gamma**2*(X[i,l]-Y[j,l])*(X[i,p]-Y[j,p])*RBF(sigma)([X[i]], [Y[j]])[0,0]
    return gram_dx #shape (n, n, d, d)



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

def dot_kernel_manual(x,y, sigma=1):
    """
    dot_f(x,y) = (x*y + sigma)**2
    """
    kernel = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            kernel[i, j] = (x[i]*(y[j]) + sigma)
    return kernel

def analytical_derivative_dot_kernel(x, y, sigma=1):
    """
    dot_k(x,y) = (x*y + sigma)**2
    dk/dx = 2*(x*y + sigma)*y
    dk/dxdx = 
    """
    derivative = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            derivative[i, j] = 2*y[j]
    return derivative

def analytical_derivative_dot_kernel_2(x, y, sigma=1):
    """
    dot_k(x,y) = (x*y + sigma)**2
    dk/dx = 2*(x*y + sigma)*y
    dk/dxdx = 2*y*y
    """
    derivative = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            derivative[i, j] = 0
    return derivative

def poly_distance_kernel_manual(x,y, sigma=1):
    """
    dot_f(x,y) = (x*y + sigma)**2
    """
    kernel = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            kernel[i, j] = (x[i] - y[j])**sigma
    return kernel

def poly_distance_derivative_manual(x, y, sigma=1):
    """
    dot_k(x,y) = (x*y + sigma)**2
    dk/dx = 2*(x*y + sigma)*y
    dk/dxdx = 
    """
    derivative = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            derivative[i, j] = sigma*(x[i] - y[j])**(sigma-1)
    return derivative

def poly_distance_derivative_2_manual(x, y, sigma=1):
    """
    dot_k(x,y) = (x*y + sigma)**2
    dk/dx = 2*(x*y + sigma)*y
    dk/dxdx = 2*y*y
    """
    derivative = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            derivative[i, j] =  sigma*(sigma-1)*(x[i] - y[j])**(sigma-2)
    return derivative

