import numpy as np
from newton import *





def funct(x:np.array):
    """Rosenbrock function"""
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0]**2) ** 2

def x1(x:np.array):
    return -400 * x[0] * (-x[0]**2 + x[1]) + 2*x[0] - 2

def x2(x:np.array):
    return -200 * x[0] ** 2 + 200 * x[1]


def x1_x1(x:np.array):
    return 1200 * x[0]**2 - 400 * x[1] + 2
def x1_x2(x:np.array):
    return -400 * x[0]
def x2_x2(x:np.array):
    return 200




if __name__ == "__main__":
    '''Main'''
    a = np.array([2, 3])
    gradient = [x1, x2]
    hessian = [[x1_x1, x1_x2],[x1_x2, x2_x2]]
    print('local newton', newton(funct, gradient, hessian, a, 0, 10000, 100, 0.0001))
    print('global newton', newton(funct, gradient, hessian, a, 1, 10000, 100, 0.0001))