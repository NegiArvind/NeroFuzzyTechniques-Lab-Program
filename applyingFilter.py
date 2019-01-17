import numpy as np
from PIL import Image
img=Image.open("tiger.jpg")
img=np.array(img)
img.transpose(2,0,1).reshape(3,-1);
img.resize(1000,1000);
np.insert(img,0,0,axis=1)
print(img)
rows=img.shape[0]; # number of rows in matrix
columns=img.shape[1];  # number of column in matrix
print(rows) 
print(columns)
myFilter=[[0,0,0],[0,1,0],[0,0,1]]




