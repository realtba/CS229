import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from math import log

#Create the Design Matrix X and Target Vector Y
Y = np.array([float(line) for line in open('q1y.dat', 'r')])
X = np.array([[1,float(line.split()[0]), float(line.split()[1])] for line in open('q1x.dat', 'r')])

#The Hypothesis h, expit is the logistic function 
def h(theta, arg):
	return expit(theta[0]*arg[0]+theta[1]*arg[1]+theta[2]*arg[2])
#The first derivative of the log likelihood
def del_L(theta, X,Y):
	# sum over all feature vectors x and corresponding target y
	return sum((y-h(theta,x))*x for x,y in zip(X,Y))
#The second derivative of the log likelihood
def del2_L(theta,X, Y):
	# sum over all feature vectors x and corresponding target y
	return sum(-h(theta, x)*(1-h(theta,x))*np.outer(x,x) for x,y in zip(X,Y) )

theta = np.zeros(shape=(3,))
max_iter = 50

# Newton Method to maximize the log likelihood
for i in range(max_iter):
	theta = theta - np.dot( np.linalg.inv(del2_L(theta,X,Y)),del_L(theta,X,Y))
#Plot the Features
plt.plot([row[1] for row in X[Y==1,:]], [row[2] for row in X[Y==1,:]], 'g^')
plt.plot([row[1] for row in X[Y==0,:]], [row[2] for row in X[Y==0,:]], 'ro')
#Plot the decision boundary (i,j) given by theta[0]]+theta[1]* i + theta[2]j = 0
plt.plot( [(-theta[0]-i*theta[1])/theta[2] for i in range(10)])
plt.show()
