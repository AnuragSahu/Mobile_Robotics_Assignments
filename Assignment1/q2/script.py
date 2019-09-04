import matplotlib.pyplot as plt
import math
from PIL import Image,ImageDraw 
import matplotlib.mlab as mlab
from numpy import linalg as LA
import numpy as np
import cv2
import math


K = np.array([[7.2153e+02,0,6.0955e+02],[0,7.2153e+02,1.7285e+02],[0,0,1]])
height_camera = 1.65
#Drawing the lines in the image
def draw_line(img,points):
	color_of_line = (22,250,7)
	line_width = 2
	cv2.line(img,(points[0][0],points[1][0]),(points[0][1],points[1][1]),color_of_line,line_width)
	cv2.line(img,(points[0][3],points[1][3]),(points[0][1],points[1][1]),color_of_line,line_width)
	cv2.line(img,(points[0][2],points[1][2]),(points[0][3],points[1][3]),color_of_line,line_width)
	cv2.line(img,(points[0][0],points[1][0]),(points[0][2],points[1][2]),color_of_line,line_width)
	cv2.line(img,(points[0][4],points[1][4]),(points[0][5],points[1][5]),color_of_line,line_width)
	cv2.line(img,(points[0][7],points[1][7]),(points[0][5],points[1][5]),color_of_line,line_width)
	cv2.line(img,(points[0][6],points[1][6]),(points[0][7],points[1][7]),color_of_line,line_width)
	cv2.line(img,(points[0][6],points[1][6]),(points[0][4],points[1][4]),color_of_line,line_width)
	cv2.line(img,(points[0][0],points[1][0]),(points[0][4],points[1][4]),color_of_line,line_width)
	cv2.line(img,(points[0][2],points[1][2]),(points[0][5],points[1][5]),color_of_line,line_width)
	cv2.line(img,(points[0][6],points[1][6]),(points[0][1],points[1][1]),color_of_line,line_width)
	cv2.line(img,(points[0][7],points[1][7]),(points[0][3],points[1][3]),color_of_line,line_width)
	
def get_projection_matrix(x,y):
	[tx,ty,tz] = np.dot(np.linalg.inv(K),np.array([x,y,1]))
	tx = (tx / ty)*height_camera
	tz = (tz / ty)*height_camera
	ty = (ty / ty)*height_camera
	c = (math.cos(-2.5*(math.pi/180)))
	s = (math.sin(-2.5*(math.pi/180)))
	Projection_matrix = np.array([[ c, 0, s,  tx],[ 0, 1, 0,1.65],[-s, 0, c,  tz]])
	Projection_matrix = np.dot(K,Projection_matrix)
	return Projection_matrix
	

img = Image.open('image.png')

x = 752
y = 260


# Giving the Physical world cordinated mapped to the points 
car_z = 4.10
car_y = 1.38 
car_x = 1.51

X = [0,car_x,0,car_x,0,0,car_x,car_x]
Y = [0,0,-car_y,-car_y,0,-car_y,0,-car_y]
Z = [0,0,0,0,-car_z,-car_z,-car_z,-car_z]
R = np.array([X,Y,Z,np.ones(8)])
Projection_matrix = get_projection_matrix(x,y)
print("The Projection_matrix is :")
print(Projection_matrix)
relative_points = np.dot(Projection_matrix,R)
# Making the last row as one and dividing the rest of the rows by last row
for i in range(3):
    for j in range(8):
        relative_points[i][j] = relative_points[i][j]/relative_points[2][j]

# Making the real points real so that they can be plotted on the image
relative_points = np.round(relative_points).astype(int)
img = np.array(img)
ax,ay,az = relative_points
plt.scatter(ax,ay)
draw_line(img,relative_points)
# Showing the plot and image
plt.imshow(img)
plt.show()
