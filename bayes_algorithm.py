import numpy as np
import pandas as pd

def load_dataset():
	dataset=pd.read_csv("bayes_input.csv")
	# dataset.iloc[np.random.permutation(len(dataset))]
	dataset = dataset.sample(frac=1).reset_index(drop=True)
	# dataset=shuffle(dataset)
	columns=len(dataset.columns)
	x=dataset.iloc[:,:-1].values
	y=dataset.iloc[:,columns-1].values
	y=y.reshape(y.shape[0],1)
	print(x.shape)
	print(y.shape)
	return x,y

train_size=0

def get_train_and_test_dataset(x,y):
	global train_size
	m=y.size
	train_size=int(m*0.7)
	x_train=x[:train_size]
	x_test=x[train_size:]
	y_train=y[:train_size]
	y_test=y[train_size:]
	print("train size ", train_size)
	return x_train,x_test,y_train,y_test

x,y=load_dataset()
print("x",x)
print("y",y)
m=y.shape[0]

x_train,x_test,y_train,y_test=get_train_and_test_dataset(x,y)
print("x_train",x_train)
print("x_test",x_test)
print("y_train",y_train)
print("y_test",y_test)

total_positive_class=np.count_nonzero(1)
total_negative_class=m - total_positive_class

def cal_conditional_prob(feature_column_index,value,class_type,x,y):
	# print("hi")
	global total_positive_class,total_negative_class,train_size
	# print(total_positive_class,total_negative_class,train_size)
	#print("conditional ",feature_column_index,value,class_type)
	count=0
	for i in range(train_size):
		if y[i]==class_type and x[i][feature_column_index]==value:
			count+=1

	if class_type==0:
		return count/total_negative_class
	else:
		return count/total_positive_class

def calculate_posterior_probability_of_yes(x_test,x,y):
	global total_positive_class,m
	res=total_positive_class/m
	#print("xshape",x_test.size)
	print(x_test[0])

	for i in range(x_test.size):
		res=res*cal_conditional_prob(i,x_test[i],1,x,y)

	return res


def calculate_posterior_probability_of_no(x_test,x,y):
	global total_negative_class,m
	res=total_negative_class/m
	# print("xshape",x_test.size)
	print(x_test[0])
	for i in range(x_test.shape[0]):
		res=res*cal_conditional_prob(i,x_test[i],0,x,y)
	return res

def check_test_set(x_test,x_train,y_train):
	print(x_test[0])
	prob_yes=calculate_posterior_probability_of_yes(x_test[0],x_train,y_train)
	prob_no=calculate_posterior_probability_of_no(x_test[0],x_train,y_train)
	print("Yes probability",prob_yes)
	print("No probability",prob_no)

	if prob_yes>prob_no:
		print("Samples belong to positive class")
	else:
		print("Sample belong to negative class")


check_test_set(x_test,x_train,y_train)
