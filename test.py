import numpy as np
import cv2 as cv

cam_pam = np.load('calibration.npz')
k = cam_pam['newCamMat']
dist = cam_pam['dist']
F_Mat = np.load('Fundamental_Matrix.npz')
F = F_Mat['F']

# Computing the focal length
focal_length = (k[0][0]+k[1][1])/2
np.savez('Focal_Length.npz', focal_length = focal_length)

# Transpose of K
k_tp = np.transpose(k)

#print('The Camera Calibration Matrix: \n', k)
#print('Transpose of Camera calibration matrix \n', k_tp)
#print('The fundamental Matrix: \n', F)

# Calculation of Essential matrix
temp = np.matmul(k_tp, F)
E = np.matmul(temp, k)
#print('The essential matrix: \n', E)

# Singular Value decomposition of E
u, s, v = np.linalg.svd(E, full_matrices=True)
#print('Values after SVD:')
#print('u: \n', u)
#print('s: \n',s)
#print('v:\n', v)

sig1 = s[0]
sig2 = s[1]
sigma = (sig1 + sig2)/2
#print('sigma value: ', sigma)
s = np.zeros((3,3))
s[0][0] = sigma
s[1][1] = sigma

#s[0] = sigma
#s[1] = sigma

# New Essential Matrix to remove errors and make sigma1 = sigma2 and sigma3 = 0
E_new = np.matmul(u, np.matmul(s,v))

print('New E:\n', E_new)
u_n, s_n, v_n = np.linalg.svd(E_new)
# print('u_new:\n', u_n)
# print('v_new:\n', v_n)
# print('s_new:\n', s_n)

# Computing the translation and rotation matrices
W = np.zeros((3, 3))
W[0][1] = -1
W[1][0] = 1
W[2][2] = 1
W_inv = np.transpose(W)

# u_t = np.transpose(u)
# v_t = np.transpose(v)
#
# x = np.matmul(W, u_t)
# x = np.matmul(s, x)
# # Translation Vector
# T = np.matmul(u, x)
# print('T: \n', T)

u_t_n = np.transpose(u_n)
v_t_n = np.transpose(v_n)

x_n = np.matmul(W, u_t_n)
x_n = np.matmul(s, x_n)
# Translation Vector
T_n = np.matmul(u_n, x_n)
#print('T: \n', T_n)

# Rotation Matrix:
R = np.matmul(W_inv, v_t_n)
R = np.matmul(u_n, R)

#print('The Rotation Matrix: \n', R)
#print('Determinant of the rotation Matrix: ', np.linalg.det(R))
T_n = -T_n
R = -R

# Generating the translation vector from the skew symmetric matrix
t = np.zeros((3,1))
t[0,0]= T_n[2,1]
t[1,0]= T_n[0,2]
t[2,0]=T_n[1,0]

#print('Shape of calculated translation vect: ', t.shape)
np.savez('Algebraic_Rot_T.npz', R = R, t= t)

#print('R*T', np.matmul(R, T_n))
#print('T*R', np.matmul(T_n,R))

# #print('T: \n', T_n)
# #print('R:\n', R)
# #print(np.linalg.det(R))
#
# Computing the camera projection matrices
# The Left Camera Projection Matrix:
I = np.identity(3)
z = np.zeros((3,1))
X = np.concatenate((I,z),axis=1)
#print(X)

# The Left camera Projection Matrix
Proj_Left = np.matmul(k, X)
print('The left Camera Projection Matrix: \n', Proj_Left)

# The Right Camera Projection Matrix
# Concatenating the Rotational matrix and the translation vector along the column
Y = np.concatenate((R,t), axis = 1)
Proj_Right = np.matmul(k,Y)
print('The Right Camera Projection Matrix: \n', Proj_Right)

# Using the pose function
inLiners = np.load('Inliners.npz')
inLeft = inLiners['inLeft']
inRight = inLiners['inRight']

# This method takes homogeneous triangulated 3-D points, the rotation vector, the translation vector, Camera Matrix,
# Distortion Coefficients as the input parameters and returns the projected points on to the image plane.
# The rotation vector is computed from the rotation matrix by using the cv2.Rodriguez method.
# triangulatedPts, Rot, Trans, Mask = cv.recoverPose(E = E_new, points1=inLeft, points2=inRight, cameraMatrix=k)
ret, Rot, Trans, Mask, triangulatedPts = cv.recoverPose(E_new, points1=inLeft, points2=inRight, cameraMatrix=k, distanceThresh = 1000)

np.savez('Rot_and_Trns.npz', Rot = Rot, Trans = Trans)

#print('The rotation Matrix from this method:\n', Rot)
print('The Translation vector from this method:\n', Trans)
#print('Det of new R:', np.linalg.det(Rot))


print('Triangulated Points: \n',triangulatedPts)

points = cv.convertPointsFromHomogeneous(np.transpose(triangulatedPts))

print('Points after converting to homogeneous:\n',points)

# Computing the error
rotVec = cv.Rodrigues(Rot)

# The following code is taken from Dr. Richard Alan Peters

# Projecting the triangulated World Points on the left image plane
proj_pts_L = cv.projectPoints(points, (0,0,0), (0,0,0), k, dist)
# Projecting the triangulated World Points on the right image plane
proj_pts_R = cv.projectPoints(points, rotVec[0],Trans, k, dist)

# Using the re-projected inlier points to the image plane and the original inlier points,
# the reprojection error is computed.
# Computing the difference between re-projected points and the original points for both the left and right images
# in both x and y directions
dLx = np.squeeze(proj_pts_L[0][:,0,0] - inLeft[:, 0, 0])
dRx = np.squeeze(proj_pts_R[0][:,0,0] - inRight[:, 0,0])
dLy = np.squeeze(proj_pts_L[0][:,0,1] - inLeft[:, 0, 1])
dRy = np.squeeze(proj_pts_R[0][:,0,1] - inRight[:, 0,1])

# To get the number of the total terms inorder to compute the average
npts = inLiners['npts']
# Computing the average error for both images in both the x and y directions
pt_error_left_x = np.sum(np.abs(dLx))/npts
pt_error_right_x = np.sum(np.abs(dRx))/npts
pt_error_left_y = np.sum(np.abs(dLy))/npts
pt_error_right_y = np.sum(np.abs(dRy))/npts

print('errors:\n','Pt_error_left_x: ',pt_error_left_x,'\n pt_error_left_y: ', pt_error_left_y,'\n pt_error_right_x: ', pt_error_right_x,'\n pt_error_right_y: ', pt_error_right_y)



