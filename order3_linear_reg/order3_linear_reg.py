# MUDITT KHURANA // UnHingedFrog
# LINEAR REGRESSION - LOSS FUNCTION = (ABS(X * THETA - Y))^3  i.e. third order
# X contains two columns, one containing ones(1's) and the other the data
# y is the dependant variable
# data as been normalized

import numpy as np
import pandas as pd
import matplotlib.pyplot as plotdata

def cube(e):                            # function to find cube of each element of matrix
    it = np.nditer([e, None])
    for a, b in it:
        b[...] = a*a*a
    return it.operands[1]

data=np.genfromtxt('sampledataset',delimiter=',')
X=data[:,0:2]
y=data[:,[2]]
#print(y)
#plotdata.scatter(X,y)
#plotdata.show()
n=90000                                 # number of iterations
m=len(X)                                # number of training data
theta=np.multiply(np.ones((1,2)),1)     # initial parameters of the best fit line
alpha=0.9                               # learning data
J=np.zeros((n,1))                       # loss function - stores the loss function after each iteration       
s=0                                     
r=0
X1=np.zeros(np.shape(X))                # partial derivatives
#X=np.multiply(X,1/10000)
for i in range(0,n):
    m_error=np.dot(X,np.transpose(theta))-y
    error=cube(m_error)
    error1=np.square(m_error)
    mi=np.sum(np.absolute((error)),axis=0)  # loss function before theta changes 
    for j in range(0,m):
        if(np.dot(X[j,:],np.transpose(theta))>=y[j]): 
            X1[j,:]=X[j,:]
        else: X1[j,:]=-X[j,:]
    theta=theta-np.transpose(np.multiply(np.dot(np.transpose(X1),error1),alpha/m))  # gradient descent
    if(i%100==0): print(mi)                 # intermediate loss value is printed after 100 iterations
    error=np.dot(X,np.transpose(theta))-y
    error=cube(error)
    r=np.sum(np.absolute((error)),axis=0)   # loss function after theta changes
    J[i]=r
    if(r<mi):                               # to find best theta
        mi=r
        theta_min=theta
print(theta_min)
y1=np.dot(X,np.transpose(theta_min))                # values predicted by the obtained the obtained theta_min 
plotdata.plot(X[:,1],y1,'r')                        # best fit line
plotdata.scatter(X[:,1],y)                          # original data
plotdata.show()
plotdata.plot(list(range(0,n)),J)                   # plot of the loss function
plotdata.show()
