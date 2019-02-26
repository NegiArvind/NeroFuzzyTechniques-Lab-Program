import numpy as np
import pandas as pd
import math
dataset=pd.read_csv("backprop_input.csv")
output_neuron=2
hidden_neuron=3
total_layer=3
columns=len(dataset.columns)
input_neuron=columns-output_neuron
print("no of input input_neuron",input_neuron)
x=dataset.iloc[:,:-output_neuron].values
y=dataset.iloc[:,input_neuron:].values
learning_rate=0.1
print("X ",x)
print("Y ",y)

weight=[[[0.1, 0.1, -0.2],
         [0.2 ,0 ,0.2],
         [0.5 ,0.3 ,-0.4]],
         [[-0.1, -0.4, 0.1, 0.6],
          [0.6, 0.2, -0.1, -0.2]]]

# length = len(sorted(weight,key=len, reverse=True)[0])
# res=np.array([xi+[None]*(length-len(xi)) for xi in x])
# print(res)
# weight=np.array(weight)
print(weight)

def activation(temp_input_array):
	temp_output_array=np.zeros(len(temp_input_array))
	temp_output_list=[]
	for i in range(len(temp_input_array)):
		value=round(1/(1+math.exp(-temp_input_array[i])),3)
		temp_output_list.append(value)
	temp_output_list.insert(0,1)
	return temp_output_list

input_array=[np.zeros(input_neuron),np.zeros(hidden_neuron),np.zeros(output_neuron)]
output_array=[np.zeros(input_neuron),np.zeros(hidden_neuron+1),np.zeros(output_neuron)]
output_array[0]=x[0]
l=len(x)
x_copy=x;
print("shaape",len(x[0]))
print("input",input_array)
for i in range(total_layer-1):
	print("iiiiiiii",i)
	# print(" weight ",weight[i]," output_array 123",output_array[i])
	input_array[i+1]=np.dot(np.array(weight[i]),np.transpose(output_array[i]))
	print("input_array ",input_array[i+1])
	temp=np.array(activation(input_array[i+1]))
	print("temp",temp)
	output_array[i+1]=temp
	print("shaoe",output_array[i+1].shape)
	print("output_array",output_array[i+1])
