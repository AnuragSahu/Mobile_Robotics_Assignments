import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


K = [[406.952636, 0.000000, 366.184147], [0.000000, 405.671292, 244.705127], [0.000000, 0.000000, 1.000000]]

ts = 0.1315
td = 0.0790
p2 = image_Coordinates = np.array([[284.56243896, 149.2925415],
					               [373.93179321, 128.26719666],
					               [387.53588867, 220.2270813],
					               [281.29962158, 241.72782898],
					               [428.86453247, 114.50731659],
					               [524.76373291, 92.09218597],
					               [568.3659668, 180.55757141],
					               [453.60995483, 205.22370911]])

p1 = world_coordinates = np.array([[0,0,1],
					 		       [ts,0,1],
					 		       [ts,ts,1],
					 		       [0, ts,1],
					 		       [ts+td,0,1],
					 		       [ts+ts+td,0,1],
					 		       [ts+ts+td,ts,1],
					 		       [ts+td,ts,1]])

A = []

for i in range(0, len(p1)):
    x, y = p1[i][0], p1[i][1]
    u, v = p2[i][0], p2[i][1]
    A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
    A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
A = np.asarray(A)
U, S, Vh = np.linalg.svd(A)
Last_row = Vh[-1,:]# / Vh[-1,-1]
homography_matrix = Last_row.reshape(3, 3)
print("The homography_matrix is :")
print(homography_matrix)

image = './image.png'
im = Image.open(image)
plt.imshow(im)
a = np.dot(homography_matrix,p1.T)
ax,ay,az = a
ax = (ax/az)
ay = (ay/az)
plt.scatter(ax,ay)

# The Bonous part of the question
k_inv = np.linalg.inv(K)
[r1,r2,t] = np.dot(k_inv,homography_matrix)
r3 = np.cross(r1,r2)
R = [r1,r2,r3]
U, S, V = np.linalg.svd(R)

R = np.array([[1,0,0],[0,1,0],[0,0,np.linalg.det(np.dot(U,V.T))]])
R = np.dot(U,R)
R = np.dot(R,V.T)
print("The R Matrix : ")
print(R)

print("The t Matrix : ")
print(R[2]/np.linalg.norm(R[0]))

plt.show()