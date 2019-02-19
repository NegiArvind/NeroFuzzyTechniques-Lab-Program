import pandas as pd
import numpy as np

dataset=pd.read_csv("input_madaline.csv")
columns=len(dataset.columns)

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,columns-1].values

rows=x.shape[0]

weight=np.full((columns-2,columns-1),0.1)
weight[0]=[0.3,0.05,0.2]
weight[1]=[0.15,0.1,0.2]

final_layer_weight=np.full(columns-1,0.1)
final_layer_weight=[0.5,0.5,0.5]
print(weight)
print("x",x)
z_output=np.full(columns-1,0)
print(z_output)
z_list=[]
learning_rate=0.5

def activation_function(x):
	if x>=0:
		return 1
	else:
		return -1

# def find_minimum(z_output):
	# min_value=0
	# for i in range(z_output.shape):
	# 	if(z_output[i]>0) and if z_output<min_value:

iteration=0
flag=True
while iteration<3:
	for j in range(rows):
		z_list=[]
		z_list_value=[]
		for i in range(columns-2):
			# print("x]j]",x[j])
			# print("transpose",np.transpose(x[j]))
			# print("weight",weight[i])
			out=np.dot(np.transpose(x[j]),weight[i])
			# print("output",out)
			z_list_value.append(out)
			z_list.append(activation_function(out))
		z_list.insert(0,1) #adding bias weight
		min_value=min(z_list,key=abs)
		min_value_index=z_list.index(min_value)

		# print(min_value_index)
		z_output=np.array(z_list)
		y_out=activation_function(np.dot(np.transpose(z_output),final_layer_weight))
		print("y_out ",y_out)

		if(y_out!=y[j] and y[j]==1):
			weight[min_value_index]=weight[min_value_index]+x[j]*(learning_rate*(y[j]-z_list_value[min_value_index]))
			# print("when 1 ",weight[min_value_index])
			flag=True
		elif(y_out!=y[j] and y[j]==-1):
			for i in range(len(z_list_value)):
				if(z_list_value[i]>0):
					weight[i]=weight[i]+x[j]*(learning_rate*(y[j]-z_list_value[min_value_index]))
			# print("when 0 weight",weight)
			flag=True
		else:
			flag=False
		print("Updated weight",weight)
		# print("output",z_output)
	iteration=iteration+1
	print("iteration ",iteration)
