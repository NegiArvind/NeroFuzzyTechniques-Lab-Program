import numpy as np
from PIL import Image
img=Image.open("emma.jpg") #used to open the image
img=np.array(img) # it converts image into numpy array
img.transpose(2,0,1).reshape(3,-1); # converting 3d numpy array into 2d numpy array
img.resize(500,500); # Resizing the given image into 500*500
img=np.insert(img,0,0,axis=1) # adding 1d array of zeroes on first column
img=np.insert(img,0,0,axis=0) # adding 1d array of zeroes on first row
img=np.insert(img,img.shape[1],0,axis=1) # adding 1d array of zeroes on last column
img=np.insert(img,img.shape[0],0,axis=0) # adding 1d array of zeroes on last row
print(img)
Image.fromarray(img).show()
rows=img.shape[0]; # number of rows in matrix
columns=img.shape[1];  # number of column in matrix
print(rows) 
print(columns)
myFilter=[[1,1,1],[1,1,1],[1,1,1]] # filters of 3*3 matrix
newImage=[]
for i in range(rows-2):
	newList=[]
	for j in range(columns-2):
		sum1=0
		for k in range(3):
			for m in range(3):
				sum1+=img[i+k,j+m]*myFilter[k][m]
		newList.append(int(sum1/9));
	newImage.append(newList);
newImage=np.array(newImage)
print(newImage.shape[0])
print(newImage.shape[1])
print(newImage)

filteredImage=Image.fromarray(newImage,'RGB')
filteredImage.save("filteredImageTiger.jpg")
filteredImage.show();



