import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

dataframe=pd.read_csv("k_means_input.csv")
x=dataframe.iloc[:,:].values
print(x)
size=x.shape[0]
sse_list=[]
for i in range(1,size):
	print("Cluster of size ",i)
	random_centroid=x[np.random.choice(range(0,size),i,replace=False)]
	print(random_centroid)
	clusters={}

	previous_centroid=[None]*i

	for j in range(0,i):
		clusters[j]=[]
	end=True
	while end:
		for j in range(0,size):
			mindis=99999999999
			index=0
			for k in range(i):
				sub=x[j]-random_centroid[k]
				dis=math.sqrt(np.sum(np.square(sub)))
				if dis<mindis:
					index=k
					mindis=dis
			clusters[index].append(j)
		for m in range(0,i):
			val=round(sum(clusters[m])/len(clusters[m]))
			random_centroid[m]=val
			if val != previous_centroid[m]:
				end=False
	sse=0;
	for j in range(0,i):
		for m in range(0,len(clusters[j])):
			sse=sse+np.square(np.sum(clusters[j][m]-random_centroid[j]))
	sse_list.append(sse)

plt.plot(np.arange(1,size),sse_list,'-')
plt.show()














