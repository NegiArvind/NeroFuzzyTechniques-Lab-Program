import pandas as pd
import numpy as np
dataset=pd.read_csv("inputAdaline.csv")
columns=len(dataset.columns)
x=dataset.iloc[:,:-1].values 
y=dataset.iloc[:,columns-1].values
print("Input: ",x)
print("Target Output: ",y)
weight=np.full(columns-1,0.1)
print("Initial weight: ",weight)
row=x.shape[0]
tolerance_error=1.4
ec=100
j=0;
learningRate=0.1
while ec>tolerance_error:
	ec=0
	for i in range(row):
		yin=np.dot(x[i],np.transpose(weight))
		diff=y[i]-yin
		weight=weight+(x[i]*(learningRate*diff))
		# print("new weight",weight)
		# print("answer",yin)
		ec=ec+(diff)*(diff);
	print("Weight after ",j+1," iteration is : ",weight)
	print("Error in ", j+1," iteration is : ",ec)
	print()
	j+=1
print("Iteration : ",j)
print("weight= ", weight)

