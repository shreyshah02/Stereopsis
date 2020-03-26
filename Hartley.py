from pylab import *
from numpy import *
from scipy import linalg
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#
# Loading the camera parameters calculated during camera calibration
cam_parameters = np.load('calibration.npz')
# Camera Matrix
k = cam_parameters['mtx']
newK = cam_parameters['newCamMat']
# print('The camera Matrix: \n', k)
# Distortion Coefficients
dist = cam_parameters['dist']

# Loading the stereo pair
left_img = cv.imread('laptop_left.JPG')
right_img = cv.imread('laptop_right.JPG')

# Shape of the left image
R, C, B = left_img.shape

# Converting the images to grayscale
gray_L = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
gray_R = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)

# Using the SIFT feature matching technique from the Assignment 2.
# More explaination of this code part has been given in the report.
# Create object of SIFT
sift = cv.xfeatures2d.SIFT_create()

# Detect the keypoints
kp_L, des_L = sift.detectAndCompute(gray_L, None)
kp_R, des_R = sift.detectAndCompute(gray_R, None)

# FLANN Parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_L, des_R,k=2)

good = []
matched_L = []
matched_R = []

for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        matched_L.append(kp_L[m.queryIdx].pt)
        matched_R.append(kp_R[m.trainIdx].pt)

# Converting the matched keypoints into an array so that it can be passed in a function to draw epilines
# that takes input as a array
matched_L = np.array(matched_L)
matched_R = np.array(matched_R)
print('Shape of matched keypoints: ', matched_L.shape)

#Criteria is defined as follows criteria = (type, number of iterations, accuracy). In this case we are telling the
# algorithm that we care both about number of iterations and accuracy (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER)
# and we selected 30 iterations and an accuracy of 1e-6
# Criteria
crit_cal = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

# Adding New axis to the matched points. This is done so that the points have a dimension of 1 x N x 2.
# This dimension is required because the next opencv function used in line 70 takes these points as the input with this
# particular dimension of 1 x N x 2
matched_L_na = matched_L[:, np.newaxis, :]
matched_R_na = matched_R[:, np.newaxis, :]

# Undistorting the matched key points
# The next step is to multiply the matched keypoints with the inverse camera matrix, undidtort the points and then
# multiply again with the camera matrix.
# In the case of lens distortion, the equations are non-linear and depend on 3 to 8 parameters (k1 to k6, p1 and p2).
# Hence, it would normally require a non-linear solving algorithm (e.g. Newton's method, Levenberg-Marquardt algorithm,
# etc) to inverse such a model and estimate the undistorted coordinates from the distorted ones.
# And this is what is used behind function undistortPoints, with tuned parameters making the optimization fast but a
# little inaccurate.
# The explaination of the function and its input arguments are given in the report
undist_pts_L = cv.undistortPointsIter(matched_L_na, k, dist, R = None, P = newK, criteria=crit_cal)
undist_pts_R = cv.undistortPointsIter(matched_R_na, k, dist, R = None, P = newK, criteria=crit_cal)
print('Shape of the returned Undistorted points: ', undist_pts_L.shape)

undist_pts_L=undist_pts_L.reshape(-1,2)
undist_pts_R=undist_pts_R.reshape(-1,2)

# The code for this fucntion is derived from https://github.com/jesolem/PCV/blob/master/pcv_book/sfm.py
# This github repository belongs to Jan Erik Solem
def compute_fundamental(x1, x2):
    """    Computes the fundamental matrix from corresponding points
        (x1,x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # build matrix for equations
    A = zeros((n, 9))
    for i in range(n):
        A[i] = [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i] * x2[2, i],
                x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i] * x2[2, i],
                x1[2, i] * x2[0, i], x1[2, i] * x2[1, i], x1[2, i] * x2[2, i]]

    # compute linear least square solution
    U, S, V = linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F
    # make rank 2 by zeroing out last singular value
    U, S, V = linalg.svd(F)
    S[2] = 0
    F = dot(U, dot(diag(S), V))

    return F / F[2, 2]

F = compute_fundamental(undist_pts_L,undist_pts_R)
print('Fundamental Matrix=',F)

E= np.matmul(np.transpose(newK), np.matmul(F,newK))
print('Old Essential Matrix:\n', E)
# Computing Singular Value Decomposition:
U,S,V = np.linalg.svd(E)
s= (S[0] + S[1])/ 2
S[0] = S[1] = s
S[2]= 0
S=np.diag(S)
E= np.matmul(U, np.matmul(S, np.transpose(V)))
print('Essential Matrix=', E)

ret, R, t, mask, TriangulatedPoints = cv.recoverPose(E,matched_L,matched_R,newK, distanceThresh=1000)
print ('rotation vector', R)

print('translation vector', t)


# SOurce : https://github.com/marktao99/python/tree/master/CVP/samples