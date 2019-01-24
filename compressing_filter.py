import numpy as np
from PIL import Image
img=Image.open("tiger.jpg")
img=np.array(img)

def rgb2gray(rgb):
    return np.dot(rgb, [0.299, 0.587, 0.114])

img=rgb2gray(img)
row=img.shape[0]
col=img.shape[1]
print(row)
print(col)
# img.resize(1200,1920);
# row=img.shape[0]
# col=img.shape[1]
# print(row)
# print(col)
Image.fromarray(img).show()

filtered_image=[]
new_row=row/3
new_col=col/3
filter=[[1,1,1],
        [1,1,1],
        [1,1,1]]
for i in range(0,row,3):
	lis=[]
	for j in range(0,col,3):
		val=0
		for k in range(3):
			for l in range(3):
				val+=img[i+k][j+l]
		lis.append(val/9)
	filtered_image.append(lis)
filtered_image=np.array(filtered_image)
print(filtered_image)
print(filtered_image.shape[0])
print(filtered_image.shape[1])

Image.fromarray(filtered_image).show()