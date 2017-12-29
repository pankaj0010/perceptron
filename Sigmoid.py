import numpy as np
class Sigmoid:
	def sigmoid(x):
		return 1/(1+np.exp(-x))
	def sigmoidGradient(x):
		return Sigmoid.sigmoid(x)*(1-Sigmoid.sigmoid(x))