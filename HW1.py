import numpy as np
import numpy.linalg as linalg

def grad_descent(guess,p):
    grad = get_grad(guess,p)
    step = get_step()
    while not conv_crit(guess,grad):
        guess -= get_step(step)*grad
        grad = get_grad(guess,p)
    return guess,obj(guess,p)

def get_step(step=.5):
    return step**2

def conv_crit(guess,grad):
    if linalg.norm(grad)<0.01:
        return True
    else:
        return False

def obj(x,p):
    #return -10**4/np.sqrt(linalg.det(2*np.pi*p['sigma'])) * np.exp(-.5*np.dot(np.dot((x-mu).T,linalg.inv(sigma)),(x-mu)))
    return .5*np.dot(np.dot(x.T,p['A']),x)-np.dot(x.T,p['b'])

def get_grad(x,p):
    #return -obj(x)*np.dot(linalg.inv(sigma),(x-mu))
    return np.dot(p['A'],x)-p['b']

if __name__=='__main__':
    A = np.array([[10,5],[5,10]])
    b = np.reshape(np.array([400,400]),(2,1))
    sigma = np.array([[10,0],[0,10]])
    mu = np.reshape(np.array([10,10]),(2,1))
    p = {'A' : A, 'b' : b, 'sigma' : sigma, 'mu' : mu}

    guess = np.reshape(np.array([0.0,0.0]),(2,1))
    print(grad_descent(guess,p))



# 1.1 done, not sure what to write up for it


# CMT site?

# 1.0 how to define convergence criterion? finite difference, num of iters, norm of grad
# 1.0 how to choose learning rate / step size? ???? no good solution

# how to make SGD converge? no clear solution
# people generally fix number of iterations

# might have large constant (essentially constant) learning parameter

# ALLOW A LOT OF DEBUGGING TIME