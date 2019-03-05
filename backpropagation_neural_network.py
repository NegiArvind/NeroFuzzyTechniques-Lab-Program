import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):	
	val=1/(1+np.exp(-z))
	return val

def sigmoid_differentiation(z):
	val=sigmoid(z)*(1-sigmoid(z))
	return val

def tangential(z):	
	val=(np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
	return val

def tangential_differentiation(z):
	val=(1-np.square(tangential(z)))
	return val


def load_dataset(output_neurons):
	dataset=pd.read_csv("back_input.csv")
	columns=len(dataset.columns)
	x=dataset.iloc[:,:-1].values
	y=dataset.iloc[:,columns-output_neurons].values
	x=x.T # transposing the input matrix x=[x(first sample), x(second sample), ...]
	y=y.reshape(output_neurons,y.shape[0]) # reshaping (m,) to (1,m)
	return x,y


def initialize_weight(input_neuron,output_neuron,num_samples):

	weight=np.random.rand(output_neuron,input_neuron)
	bias_weight=np.zeros((output_neuron,num_samples))
	print((weight,bias_weight))
	return weight,bias_weight


def apply_alogrithm(y,x,num_samples,learning_rate,max_iterations):
	
	w1,b1=initialize_weight(input_neuron=2,output_neuron=2,num_samples=num_samples) # initializing weight for first layer
	w2,b2=initialize_weight(input_neuron=2,output_neuron=1,num_samples=num_samples) # initializing weight for second layer

	print("\nw1= ",w1)
	print("\nb1= ",b1)
	print("\nw2= ",w2)
	print("\nb2= ",b2)
	print("\n x = ",x)
	print("\n y = ",y)

	costs=[]

	for i in range(max_iterations):
	
		# forward propagation

		#First layer
		z1=np.dot(w1,x)+b1 # output of first layer
		a1=sigmoid(z1) # output of first layer after applying activation function

		#Second layer
		z2=np.dot(w2,a1)+b2 # output of first layer
		a2=sigmoid(z2) # output of first layer after applying activation function

		# print("\na1 = ",a1)
		print("\na2 output = ",a2)

		#Backward propagation

		da2=-(np.divide(y,a2)-np.divide(1-y,1-a2)) # calculating da which is equal to d(loss)/da

		# print("\n da2 = ",da2)

		# cost function of neural network i.e cost= -ylog(a)-(1-y)log(a)
		cost = (-1 / num_samples) * np.sum(np.multiply(y,np.log(a2.T)) + np.multiply(1-y,np.log((1-a2).T))) 


		## second layer
		dz2=da2 * sigmoid_differentiation(z2) #calculating difference error i.e dz=da.g(z)
		dw2=(np.dot(dz2,a1.T))/num_samples # calculating dw=dz.a'/m
		db2=np.sum(dz2,axis=1,keepdims=True)/num_samples # caluclating db=dz
		da1=np.dot(w2.T,dz2) 

		#first layer
		dz1=da1*sigmoid_differentiation(z1)
		dw1=(np.dot(dz1,x.T))/num_samples
		db1=np.sum(dz1,axis=1,keepdims=True)/num_samples
		da0=np.dot(w1.T,dz1)

		# weight updation of first layer
		w1=w1-dw1*learning_rate #updating the weight of first layer w1
		b1=b1-db1*learning_rate #updating the weight of first layer b1

		print("dw1 ", dw1)
		# weight updation of second layer
		w2=w2-dw2*learning_rate #updating the weight of second layer w2
		b2=b2-db2*learning_rate #updating the weight of second layer b2

		print("dw2 ", dw2)

		print("\nError in ",i+1," iteration ",cost)
		costs.append(cost)

	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per tens)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()

x,y=load_dataset(output_neurons=1)
apply_alogrithm(y,x,num_samples=x.shape[1],learning_rate=0.07,max_iterations=100)
