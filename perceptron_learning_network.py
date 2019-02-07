import numpy as np
import pandas as pd

dataset=pd.read_csv("perceptron_input.csv")
columns=len(dataset.columns)

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,columns-1].values
rows=X.shape[0]

print(X)
print(Y)
theta=0.1


def activation(yin):
	if yin>=0:
		return 1
	else:
		return 0

weight=np.full(columns-1,0.1)
weight[0]=0.3
weight[1]=0.1
weight[2]=0.1
iteration=0
print("Initial Weight : ",weight)
print()
	
while True:
	flag=False
	for i in range(rows):
		print("inside weight : ",weight,"X[i]: ",np.transpose(X[i]))
		y_in=np.dot(weight,np.transpose(X[i]))
		y_out=activation(round(y_in,3))
		print("yout",y_in,y_out,i)
		diff=(Y[i]-y_out)
		if diff!=0:
			print(diff)
			weight+=X[i]*theta*diff
			flag=True
	iteration+=1
	print("Iteration : ",iteration)
	print("Weight : ",weight)
	print()
	if flag==False:
		break;





