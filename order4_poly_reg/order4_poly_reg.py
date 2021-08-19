# MUDITT KHURANA // UnHingedFrog
# POLYNOMIAL REGRESSION (2ND DEGREE POLYNOMIAL) - LOSS FUNCTION = (ABS(X * THETA - Y))^4  i.e. fourth order
# X contains two columns, one containing ones(1's) and the other the data
# y is the dependant variable
# data as been normalized

import numpy as np
import pandas as pd
import matplotlib.pyplot as plotdata

def cube(e):
    it = np.nditer([e, None])
    for a, b in it:
        b[...] = a*a*a
    return it.operands[1]

data=np.genfromtxt('D:\machine learning\population.txt',delimiter=',')
x=data[:,0:2]
y=data[:,[2]]
#plotdata.scatter(X,y)
#plotdata.show()
n=90000                                 # number of iterations
m=len(x)                                # number of training data
X=np.zeros((m,3))                       # constructing X by adding x^2 term
X[:,0]=(x[:,0])
X[:,1]=(x[:,1])
X[:,2]=np.square(x[:,1])
theta=np.ones((1,3))                    # initial parameters of the best fit line
alpha=0.09                              # learning rate
J=np.zeros((n,1))                       # loss function - stores the loss function after each iteration      
s=0
r=0
X1=np.zeros(np.shape(X))                # partial derivatives
#X=np.multiply(X,1/10000)
for i in range(0,n):
    error=np.dot(X,np.transpose(theta))-y
    error1=cube(error)
    error=np.square(np.square(error))
    mi=np.sum((error),axis=0)           # loss function before theta changes
    u=np.transpose(np.multiply(np.dot(np.transpose(X),error1),alpha/m))
    theta=theta-np.transpose(np.multiply(np.dot(np.transpose(X),error1),alpha/m))   # gradient descent   
    if(i%100==0): print(mi)             # intermediate loss value is printed after 100 iterations
    error=np.dot(X,np.transpose(theta))-y
    error=np.square(np.square(error))
    r=np.sum((error),axis=0)            # loss function after theta changes
    J[i]=r
    if(r<mi):                           # to find best theta
        mi=r
        theta_min=theta
print(theta_min)
y1=np.dot(X,np.transpose(theta_min))                # values predicted by the obtained the obtained theta_min 
plotdata.plot(X[:,1],y1,'r')                        # best fit line
plotdata.scatter(X[:,1],y)                          # original data
plotdata.show()
plotdata.plot(list(range(0,n)),J)                   # plot of the cost function
plotdata.show()