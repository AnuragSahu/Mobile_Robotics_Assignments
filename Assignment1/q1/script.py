import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_name = './image.png'

K = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02],
	          [0.000000e+00, 7.215377e+02, 1.728540e+02], 
	          [0.000000e+00, 0.000000e+00, 1.000000e+00]])

R = np.array([[0, -1, 0, 0.06],
	 		  [0, 0, -1,-0.08],
	 		  [1, 0, 0, -0.27]])


def load_velodyne_points(points_path):
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
    points = points[:,:3]                # exclude reflectance values, becomes [X Y Z]
    points = points[::5,:]              # remove every 5th point for display speed (optional)
    points = points[(points[:,0] >= 5)]   # remove all points behind image plane (approximate)
    return points
    
if __name__ == '__main__':
    points = load_velodyne_points('lidar-points.bin')

    im = Image.open('image.png')
    plt.imshow(im)
    x,y,z = (points.T)
    points = points.T
    one = np.ones(len(x))
    points = np.vstack([points,one])
    P = K.dot(R)
    sctr = P.dot(points)
    x,y,z = sctr
    x = x/z
    y = y/z
    plt.xlim((0,1243))
    plt.ylim((350,0))
    plt.scatter(x,y,c=10000/z,s=6)
    plt.show()