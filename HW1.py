import numpy as np
import numpy.linalg as linalg

def grad_descent(guess):
    grad = get_grad(guess)
    step = get_step()
    while not conv_crit(guess,grad):
        guess -= get_step(step)*grad
        grad = get_grad(guess)
    return guess,obj(guess)

def get_step(step=.5):
    return step**2

def conv_crit(guess,grad):
    if linalg.norm(grad)<0.01:
        return True
    else:
        return False

def obj(x):
    sigma = np.array([[10,0],[0,10]])
    mu = np.reshape(np.array([10,10]),(2,1))
    return -10**4/np.sqrt(linalg.det(2*np.pi*sigma)) * np.exp(-.5*np.dot(np.dot(np.transpose((x-mu)),linalg.inv(sigma)),(x-mu)))

def get_grad(x):
    sigma = np.array([[10,0],[0,10]])
    mu = np.reshape(np.array([10,10]),(2,1))
    return -obj(x)*np.dot(linalg.inv(sigma),(x-mu))


guess = np.reshape(np.array([0.0,0.0]),(2,1))
print(grad_descent(guess))




# specific questions:

# CMT site?

# 1.0 how to define convergence criterion? finite difference, num of iters, norm of grad
# 1.0 how to choose learning rate / step size? ???? no good solution

# how to make SGD converge? no clear solution
# people generally fix number of iterations

# might have large constant (essentially constant) learning parameter

# ALLOW A LOT OF DEBUGGING TIME