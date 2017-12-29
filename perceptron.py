# Classifying data points for a linearly separable data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Sigmoid import Sigmoid as activation
from GradientDescent import GradientDescent as optimize

# Initialization
data=np.array(pd.read_csv('modifiedIrisData.csv', header=None))
x=data[:,:-1]
d=data[:,-1].reshape(len(x),1)	# Ground truth
x=np.c_[np.ones(len(x)), x]		# To accomodate the bias
w=np.random.rand(1,5)
wcopy=w

# Data visualization

color_class=[]
for i in range(data.shape[0]):
	if d[i]==0:
		color_class.append('red')
	else:
		color_class.append('green')
plt.scatter(x[0:99,2],x[0:99,3],color='red',label='Class 1')
plt.scatter(x[100:149,2],x[100:149,3],color='green',label='Class 2')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()

# Hyperparameters
iterations=500	
alpha=0.2

plotErrorCurve=True

# Optimization using Batch (Vanilla) Gradient Descent
[w1,graph_x1,graph_y1]=optimize.GD(x,w,d,alpha,iterations,plotErrorCurve)

# Optimization using Stochastic Gradient Descent
[w2,graph_x2,graph_y2]=optimize.SGD(x,w,d,alpha,iterations,plotErrorCurve)

# Error v Iterations curve for GD and SGD
if(plotErrorCurve):
	marker_size=4
	plt.subplot(2,1,1)
	plt.scatter(graph_x1, graph_y1,marker_size)
	plt.xlabel("Iterations")
	plt.ylabel("Error")
	plt.title("Batch (Vanilla) Gradient Descent")
	plt.xlim(0,iterations)
	plt.ylim(0)

	plt.subplot(2,1,2)
	plt.scatter(graph_x2, graph_y2,marker_size)
	plt.xlabel("Iterations")
	plt.ylabel("Error")
	plt.title("Stochastic Gradient Descent")
	plt.xlim(0,iterations)
	plt.ylim(0)

	plt.show()

# To see the values using trained weight (for cross-checking)
y1=activation.sigmoid(np.dot(x,w1.transpose()))
y2=activation.sigmoid(np.dot(x,w2.transpose()))
np.set_printoptions(precision=2)
print("Batch:",y1)
print("Stochastic",y2)
