import numpy as np
import numpy.linalg as LA
import numpy.random as random
import time

def grad_descent(guess,p):
    itnum = 0
    grad = get_grad(guess,p,itnum,method='symbolic')
    step = get_step()
    prevguess = [None]
    while not conv_crit(guess,prevguess[0],p):
        itnum += 1
        prevguess.append(np.copy(guess))
        if itnum > 10**3:
            prevguess.pop(0)
        guess -= get_step(step)*grad
        grad = get_grad(guess,p,itnum,method='symbolic')
    return guess,obj(guess,p)

def get_step(step=10**-6):
    return 10**-6

def conv_crit(guess,prevguess,p):
    if prevguess is not None:
        if abs(obj(guess,p)-obj(prevguess,p))<10**-2:
            return True
    else:
        return False

def obj(w,p):
    #return -10**4/np.sqrt(LA.det(2*np.pi*p['sigma'])) * np.exp(-.5*LA.multi_dot(((w-mu).T,LA.inv(sigma),(w-mu))))
    #return .5*np.dot(np.dot(w.T,p['A']),w)-np.dot(w.T,p['b'])
    WXY = np.dot(w,p['x'].T)-y
    return np.vdot(WXY,WXY)
    #i = np.ceil(100*random.rand(1))
    #return (np.dot(p['x'][i,:],w)-p['y'][i])**2

def get_grad(w,p,itnum,method='numerical'):
    if method == 'symbolic':
        #WXY = np.dot(w,p['x'].T)-p['y']
        #return np.array([np.vdot(WXY,p['x'][:,i]) for i in range(10)])
        i = itnum%100
        return (np.vdot(p['x'][i,:],w)-p['y'][i])*p['x'][i,:]
        #return -obj(w)*np.dot(LA.inv(sigma),(w-mu))
        #return np.dot(p['A'],w)-p['b']
    elif method == 'numerical':
        return (obj(w,p)-obj(w-p['delta'],p))/p['delta']

def get_ans(p):
    return LA.multi_dot((LA.inv(np.dot(p['x'].T,p['x'])),p['x'].T,p['y']))

if __name__=='__main__':
    #A = np.array([[10,5],[5,10]])
    #b = np.reshape(np.array([400,400]),(2,1))
    #sigma = np.array([[10,0],[0,10]])
    #mu = np.reshape(np.array([10,10]),(2,1))
    t0=time.clock()
    delta = 0.05
    x = np.loadtxt('fittingdatap1_x.txt')
    y = np.loadtxt('fittingdatap1_y.txt')
    p = {'delta' : delta, 'x' : x, 'y' : y}
    #p = {'A' : A, 'b' : b, 'sigma' : sigma, 'mu' : mu, 'delta' : delta}
    guess = np.zeros((10,))
    #guess = np.reshape(np.array([0.0,0.0]),(2,1))

    ans = get_ans(p)
    #print(ans)
    print(obj(ans,p))

    coord, objans = grad_descent(guess,p)
    #print(coord)
    print(objans)
    print(time.clock()-t0)






# CMT site?

# might have large constant (essentially constant) learning parameter

# ALLOW A LOT OF DEBUGGING TIME