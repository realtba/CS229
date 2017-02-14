import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from math import log
import math

#Create the Design Matrix X and Target Vector Y
Y = np.array([float(line) for line in open('q2y.dat', 'r')])
X = np.array([[1, float(line)] for line in open('q2x.dat', 'r')])

#theta given by the normal equation for weighted linear regression
def ThetaWeightedReg(x, tau): 
	W = np.diag([math.exp(-np.dot(x-X[i,1],x- X[i,1])/(2*tau*tau)) for i in range(100) ])
	return np.dot( np.linalg.inv( np.dot(np.transpose(X), np.dot(W,X))),np.dot(np.dot(np.transpose(X),W),Y))
#Plot the data

#theta given by normal equation for lienar regression
theta = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)),np.dot(np.transpose(X),Y))
#Plot the linear regression 
plt.plot([x for x in np.linspace(min(X[:-1,1]),max(X[:-1,1]),100)],[theta[0]+x*theta[1] for x in np.linspace(min(X[:-1,1]),max(X[:-1,1]),100)])
#Plot the weighted linear regression
plt.plot([x for x in np.linspace(min(X[:-1,1]),max(X[:-1,1]),100)],[ThetaWeightedReg(x,0.1)[0]+x*ThetaWeightedReg(x,0.1)[1] for x in np.linspace(min(X[:-1,1]),max(X[:-1,1]),100)], 'ro')
plt.plot([x for x in np.linspace(min(X[:-1,1]),max(X[:-1,1]),100)],[ThetaWeightedReg(x,2)[0]+x*ThetaWeightedReg(x,2)[1] for x in np.linspace(min(X[:-1,1]),max(X[:-1,1]),100)], 'g')

#Plot the data
plt.plot(X[:-1,1], Y[:-1], 'g^')
plt.show()
