import numpy as np
import pandas as pd

def sigmoid(z):
	
	val=1/1+exp(-z)
	return val

def load_dataset():
	dataset=pd.read_csv("back_input.csv")
	columns=len(dataset.columns)
	x=dataset.iloc[:,:-1].values
	y=dataset.iloc[:,columns-1].values
	print(x.shape)
	return x,y

def linear_activation_backward(da,z,activation,m):
	if activation=="relu":


	elif activation=="sigmoid":
		dz=da*sigmoid(z)
		dw=np.dot(dz,)

def initialize_weight():

	w1=np.random.rand((2,2))
	b1=np.random.rand((2,4))

	w2=np.random.rand((1,2))
	b2=np.random.rand((1,4))

	return w1,b1,w2,b2


def apply_alogrithm(x,m,learning_rate,max_iterations):
	
	w1,b1,w2,b2=initialize_weight()
	x=x.T
	costs=[]
	for i in range(max_iterations):
	
		# forward propagation

		z1=np.dot(w1,x)+b1
		a1=sigmoid(z1)

		z2=np.dot(w2,a1)+b2
		a2=sigmoid(z2)

		#Backward propagation

		da2=-(np.divide(y,a2)-np.divide(1-y,1-a2)

		cost = (-1 / m) * np.sum(np.multiply(y, np.log(a2)) + np.multiply(1 - y, np.log(1 - a2)))

		dz2=da2*sigmoid(z2)
		dw2=(np.dot(dz2,a1.T))/m
		db2=np.sum(dz2,asix=1,keepdims=True)
		da1=np.dot(w2.T,dz2)


		dz1=da1*sigmoid(z1)
		dw1=(np.dot(dz1,x))/m
		db1=np.sum(dz1,asix=1,keepdims=True)
		da0=np.dot(w1.T,dz1)

		# weight updation of first layer
		w1=w1-dw1*learning_rate
		b1=b1-db1*learning_rate

		# weight updation of second layer
		w2=w2-dw2*learning_rate
		b2=b2-db2*learning_rate

		print("Error in ",i+1," iteration ",cost)
		costs.append(cost)

x,y=load_dataset()
apply_alogrithm(x,m=4,learning_rate=0.02,max_iterations=100)





