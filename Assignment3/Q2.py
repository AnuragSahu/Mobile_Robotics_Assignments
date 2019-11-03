import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def read_file(file_name):
    pcd = o3d.io.read_point_cloud("./out_points_1.ply") # Read the point cloud
    points = np.asarray(pcd.points)
    number_of_points = points.shape[0]
    colors = np.asarray(pcd.colors)
    cloud_points=np.hstack((points,np.ones((number_of_points,1))))
    return number_of_points,cloud_points,colors

def main():
    error = 0.0001
    max_iteration = 100
    learning_rate = 0.6
    er_plot = []
    P = np.array([[-9.098548e-01, 5.445376e-02, -4.113381e-01, -1.872835e+02],
                  [ 4.117828e-02, 9.983072e-01, 4.107410e-02, 1.870218e+0],
                  [ 4.128785e-01, 2.043327e-02, -9.105569e-01, 5.417085e+01]])
    scale_o = P[2][3]
    file_name = './out_points_1.ply'
    number_of_points,cloud_points,colors = read_file(file_name)
    image_points=np.dot(P,cloud_points.T).T
    image_points_h=image_points/image_points[2]
    P0 = np.hstack((12*np.eye(3),[[4],[5],[6]])) # Initialization with Random P
    while (max_iteration):
        max_iteration-=1

        ####### Computing Jacobian_matrix ########
        image_points=np.dot(P0,cloud_points.T).T
        J=[]
        for i in range(image_points.shape[0]):
            J.append([-cloud_points[i][0]/image_points[i][2] , -cloud_points[i][1]/image_points[i][2] ,-cloud_points[i][2]/image_points[i][2] ,-1/image_points[i][2],
                        0,0,0,0,cloud_points[i][0]*image_points[i][0]/(image_points[i][2]**2),cloud_points[i][1]*image_points[i][0]/(image_points[i][2]**2),cloud_points[i][2]*image_points[i][0]/(image_points[i][2]**2),
                        image_points[i][0]/(image_points[i][2]**2)])
            J.append([0,0,0,0,-cloud_points[i][0]/image_points[i][2] , -cloud_points[i][1]/image_points[i][2] ,-cloud_points[i][2]/image_points[i][2] ,-1/image_points[i][2],
                        cloud_points[i][0]*image_points[i][1]/(image_points[i][2]**2),cloud_points[i][1]*image_points[i][1]/(image_points[i][2]**2),cloud_points[i][2]*image_points[i][1]/(image_points[i][2]**2),
                        image_points[i][1]/(image_points[i][2]**2)])
        Jacobian_matrix = np.array(J)

        ########## Finding the Residual matrix ###########
        image_points=np.dot(P0,cloud_points.T).T
        difference=[]
        for i in range(image_points.shape[0]):
            difference.append(image_points_h[i][0]-(image_points[i][0]/image_points[i][2]) )
            difference.append(image_points_h[i][1]-(image_points[i][1]/image_points[i][2]) )

        residual = np.array(difference)
        ############# Checking for the termination  ##########
        if(np.linalg.norm(Jacobian_matrix.T.dot(residual))<error):
            break;

        ############## Updating the P matrix ##########
        pseudo_inv_J = np.linalg.inv(Jacobian_matrix.T.dot(Jacobian_matrix)).dot(Jacobian_matrix.T)
        residual = residual.reshape(2*number_of_points,1)
        P0 = (P0.reshape(12,1) - (learning_rate)*np.dot(pseudo_inv_J,residual)).reshape(3,4)

        ############# Calculate the error ################
        estimated_points=np.dot(P0,cloud_points.T).T
        estimated_points = estimated_points / estimated_points[2]
        err=0;
        for i in range(cloud_points.shape[0]):
            err+=(estimated_points[i][0]-image_points_h[i][0])**2+(estimated_points[i][1]-image_points_h[i][1])**2
        er_plot.append(err)

        print("iteration left : ",max_iteration,", Error : ",err)

    plt.plot(er_plot)
    plt.show()

if __name__ == "__main__":
    main()
