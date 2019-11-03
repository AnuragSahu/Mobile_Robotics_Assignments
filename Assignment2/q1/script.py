import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_1_points = np.array([[381, 402, 1],
 						 [452, 497, 1],
 						 [671, 538, 1],
 						 [501, 254, 1],
 						 [506, 381, 1],
 						 [474, 440, 1],
 						 [471, 537, 1],
 						 [498, 364, 1],
 						 [706, 319, 1],
 						 [635, 367, 1]])

img_2_points = np.array([[390, 346, 1],
 						 [439, 412, 1],
 						 [651, 417, 1],
 						 [477, 194, 1],
 						 [482, 300, 1],
 						 [456, 359, 1],
 						 [454, 444, 1],
 						 [475, 287, 1],
 						 [686, 185, 1],
 						 [606, 253, 1]])

F = np.array([[-1.29750186e-06,  8.07894025e-07,  1.84071967e-03],
			  [3.54098411e-06,  1.05620725e-06, -8.90168709e-03],
			  [-3.29878312e-03,  5.14822628e-03,  1.00000000e+00]])

fig,(ax,bx) = plt.subplots(2,1,sharey=True)
img_1 = plt.imread('./img1.jpg')
img_2 = plt.imread('./img2.jpg')
ax.imshow(img_1)
im1x,im1y,im1z = img_1_points.T
im2x,im2y,im2z = img_2_points.T

epl1x,epl1y,epl1z = (F.T).dot(img_2_points.T)
epl2x,epl2y,epl2z = (F).dot(img_1_points.T)

for i in range(len(epl1x)):
	x = np.linspace(0,1270,1000) #im1x[i]
	y = epl1x[i]/-epl1y[i] * x + epl1z[i]/-epl1y[i]
	ax.plot(x,y,'-r')

for i in range(len(epl2x)):
	x = np.linspace(0,1270,1000)#im2x[i] #
	y = epl2x[i]/-epl2y[i] * x + epl2z[i]/-epl2y[i]
	bx.plot(x,y,'-r')

ax.scatter(im1x,im1y,c='y')
bx.scatter(im2x,im2y,c='y')
bx.imshow(img_2)
plt.show()