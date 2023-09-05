import numpy as np
from numpy import linalg as LA
import types
import random as RA


def wolfe_powell(funct:types.FunctionType, gradient, x:np.array, d:np.array, t_init:float,  omega:float, rho:float):   # Algo 6.2 (Geiger)
    """This function returns parameter satisfying the conditions of Wolfe-Powell"""
    #Phase A
    t = t_init
    gamma = 1.1
    i = 0
    a, b = 0, 0
    grad = gip(gradient, x) @ d

    while funct(x + t * d) >= funct(x) + omega * t * grad or gip(gradient, x + t * d) @ d < rho * grad:
        if funct(x + t * d) >= funct(x) + omega * t * grad:
            a = 0
            b = t


            #Phase B
            tau1 = RA.uniform(0, 0.5)
            tau2 = RA.uniform(0, 0.5)
            a_j = a
            b_j = b
            j = 0
            t = RA.uniform(a_j + tau1 * (b_j - a_j), b_j - tau2 * (b_j - a_j))

            while funct(x + t * d) >= funct(x) + omega * t * gip(gradient, x) @ d or gip(gradient, x + t * d) @ d < rho * gip(gradient, x) @ d:
                if funct(x + t * d) >= funct(x) + omega * t * gip(gradient, x) @ d:
                    b_j = t
                    j += 1
                    t = RA.uniform(a_j + tau1 * (b_j - a_j), b_j - tau2 * (b_j - a_j))
                elif funct(x + t * d) < funct(x) + omega * t * gip(gradient, x) @ d and gip(gradient, x + t * d) @ d < rho * gip(gradient, x) @ d:
                    a_j = t
                    j += 1
                    t = RA.uniform(a_j + tau1 * (b_j - a_j), b_j - tau2 * (b_j - a_j))
            return t

        elif funct(x + t * d) < funct(x) + omega * t * gip(gradient, x) @ d and gip(gradient, x + t * d) @ d < rho * gip(gradient, x) @ d:
            t = gamma * t
            i += 1

    return t


def armijo(f:types.FunctionType, x:np.array, d:np.array, gradient:list, sigma:float, beta:float):
    """This function returns parameter satisfying the conditions of Armijo"""
    t = 1
    grad = gip(gradient, x) @ d
    while f(x + t * d) - f(x) > sigma * t * grad:
        t *= beta
    return t


def gip(gradient, point): #gradient in point
    """This function returns value of gradient at the give point"""
    return np.array([f(point) for f in gradient])


def hip(hessian, point):
    """This function returns value of hessian at the give point"""
    return np.array([[f(point) for f in hessian[i]] for i in range(len(hessian))])


def is_inv(M:np.array):
    """This function returns True if matrix invertible, otherwise false"""
    return not np.isclose(LA.det(M), 0)



#algo 9.2 - 9.3 (Geiger)
def newton(funct:types.FunctionType, gradient:list,hessian:list,xinit:np.array, vers:int, maxit:int, M:float,epsilon:float, sigma:float = 0.4, rho:float = 0.5, beta = 0.5):
    """Implementation of Newton method(local(vers = 0) and global(vers = 1))"""
    x = xinit
    k = 0
    d = np.zeros_like(xinit)

    if vers == 0:    #local newton
        while LA.norm(gip(gradient, x)) > epsilon and k < maxit and LA.norm(x) < M and is_inv(hip(hessian, x)):
            d = LA.inv(hip(hessian, x)) @ (-gip(gradient, x))
            x = x + d
            k += 1

    elif vers == 1:
        t = 0.001
        while LA.norm(gip(gradient, x)) > epsilon and k < maxit and LA.norm(x) < M:
            if is_inv(hip(hessian, x)):
                d = LA.inv(hip(hessian, x)) @ (-gip(gradient, x))
                if gip(gradient, x) @ d <= -rho * LA.norm(d):
                    d = -gip(gradient, x) / LA.norm(gip(gradient, x))
            else:
                d = -gip(gradient, x) / LA.norm(gip(gradient, x))

            t = armijo(funct, x, d, gradient, sigma, beta)
            # t = wolfe_powell(funct,gradient,x,d,t,sigma,rho)
            x = x + t * d
            k += 1


    if LA.norm(gip(gradient, x)) <= epsilon:
        return [x, 'Case 0']
    elif k >= maxit:
        return [x, 'Case 1']
    elif LA.norm(x) >= M:
        return [x, 'Case 2']
    elif not is_inv(hip(hessian, x)):
        return 'case 3'









