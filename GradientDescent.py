import numpy as np
import random
import matplotlib.pyplot as plt
from Sigmoid import Sigmoid as activation
class GradientDescent:
	#Vanilla Gradient Descent
	def GD(x,w,d,alpha,iterations,plot=False):
		m=len(x)
		graph_x=np.empty(iterations)
		graph_y=np.empty(iterations)
		for i in range(iterations):
			h=np.dot(x,w.T)
			y=activation.sigmoid(h)
			J=(y-d)*(y-d)/2
			loss=sum(J)/m
			# print(loss)
			JGrad=(y-d)*activation.sigmoidGradient(h)*x
			w=w-(alpha/m)*sum(JGrad)
			if(plot):
				graph_x[i]=i
				graph_y[i]=loss
		return [w,graph_x,graph_y]

	def SGD(x,w,d,alpha,iterations,plot=False):
		m=len(x)
		graph_x=np.empty(iterations)
		graph_y=np.empty(iterations)
		for i in range(iterations):
			idx=random.randint(0,m-1)
			h=np.dot(x,w.T)
			y=activation.sigmoid(h)
			J=(y-d)*(y-d)/2
			loss=sum(J)/m
			# print(loss)
			JGrad=(y[idx]-d[idx])*activation.sigmoidGradient(h[idx])*x[idx]
			w=w-alpha*JGrad
			if(plot):
				graph_x[i]=i
				graph_y[i]=loss
		return [w,graph_x,graph_y]
