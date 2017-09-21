import numpy as np
import numpy.linalg as LA
import numpy.random as random
import time

def finiteDif(w,itnum,obj,delta):
    return (obj(w)-obj(w-delta))/delta

def grad_descent(w,obj,grad='finite_dif',delta=0.05):
    if grad == 'finite_dif':
        def grad(w,itnum): return finiteDif(w,itnum,obj,delta)
    itnum = 0
    gradval = grad(w,itnum)
    step = get_step()
    prevw = [None]
    while not conv_crit(w,prevw[0],obj):
        itnum += 1
        prevw.append(w)
        if itnum > 10.0**4:
            prevw.pop(0)
        w -= get_step(step)*gradval
        gradval = grad(w,itnum)
    return w, obj(w)

def get_step(step=10**-5):
    return 0.5*step

def conv_crit(w,prevguess,obj):
    if prevguess is not None:
        if abs((obj(w)-obj(prevguess))/obj(w))<10.0**-6:
            return True
    else:
        return False


def negGauss(w,mu,sigma): 
    # highly dependent on step size when using numerical method (step=0.5, then step**2)
    def obj(w):
        return -10**4/np.sqrt(LA.det(2*np.pi*sigma)) * np.exp(-.5*LA.multi_dot(((w-mu),LA.inv(sigma),(w-mu))))
    def grad(w,itnum):
        return -obj(w)*np.dot(LA.inv(sigma),(w-mu))
    return grad_descent(w,obj,grad)

def quadBowl(w,A,b):
    # use step = .1, step = step**1.5
    def obj(w):
        return .5*LA.multi_dot((w,A,w))-np.dot(w.T,b)
    def grad(w,itnum):
        return np.dot(A,w)-b
    return grad_descent(w,obj,grad)

def dataSet(w,x,y):
    def obj(w):
        WXY = np.dot(w,x.T)-y
        return np.vdot(WXY,WXY)
    def grad_batch(w,itnum):
        # any sufficiently small step
        WXY = np.dot(w,x.T)-y
        return np.array([np.vdot(WXY,x[:,i]) for i in range(10)])
    def grad_stochastic(w,itnum):
        # use step = 10**-5
        i = itnum%100
        return (np.vdot(x[i,:],w)-y[i])*x[i,:]
    return grad_descent(w,obj,grad_batch)



def get_ans(x,y):
    return LA.multi_dot((LA.inv(np.dot(x.T,x)),x.T,y))





if __name__=='__main__':
    #t0=time.clock()
    
    sigma = np.array([[10,0],[0,10]])
    mu = np.array([10,10])
    A = np.array([[10,5],[5,10]])
    b = np.array([400,400])
    x = np.loadtxt('fittingdatap1_x.txt')
    y = np.loadtxt('fittingdatap1_y.txt')
    w = np.zeros((2,))
    w = np.zeros((10,))
    coord,objans = dataSet(w,x,y)
    print(objans)

    #data = np.loadtxt('curvefittingp2.txt')
    #p = {'x' : data[0,:], 'y' : data[1,:], 'm' : 0}
    #print(get_ans(p))
    #print(time.clock()-t0)


    #print(get_ML(p))





# CMT site?

# might have large constant (essentially constant) learning parameter

# ALLOW A LOT OF DEBUGGING TIME