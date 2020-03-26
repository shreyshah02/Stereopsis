import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import os

# This code is derived from the code given by Dr. Peters in the lecture slides for camera calibration
# Define the chess board rows and columns
# For the interior corners of the checkerboard
# The chessboard used was 10 x 8, Hence the inner corners: 9,7
rows = 9
cols = 7

# Set the termination criteria for the corner sub-pixel algorithm
# Sets the maximum number of iterations to 30, accuracy to 0.001
criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 30, 0.001)

# Important input data needed for camera calibration is a set of 3D real world points and corresponding 2D image points.
# These image points are locations where two black squares touch each other in chess boards
# 3D points are called object points and 2D image points are called image points.
# Prepare the object points:  (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0).
# They are the same for all images
objPnts = np.zeros((rows* cols,3), np.float32)
objPnts[:,:2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

# Create the arrays to store the object points and the image points
objPntsArray = []
imgPntsArray = []

# Creating an image window
cv.namedWindow('Image', cv.WINDOW_NORMAL)

# Loop over the image files of the chessboard
# The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell
# Using glob to read the path
for path in glob.glob('Cam Calibrate Iphone\\*.jpg'):
    # Load the image and convert it to gray scale
    img = cv.imread(path)
    # Converting the images to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # findChessboardCorners - Finds the positions of internal corners of the chessboard.
    # parameter1 - source image
    # parameter2 - number of inner board corners per a chessboard row and column
    ret, corners = cv.findChessboardCorners(gray, (rows, cols), None)

    # Saving the chessboard images marked with corners found above
    cv.imwrite(path + 'corners.jpg', img)

    # Make sure the chess board pattern was found in the image
    # If found, add object points, image points (after refining them)
    if ret:
        # Refine the corner position
        # cornerSubPix determines the position of the the corners to sub pixel accuracy
        corners2 = cv.cornerSubPix(gray, corners, (11, 11),(-1, -1),criteria)

        # Adding the object points and image points
        objPntsArray.append(objPnts)
        imgPntsArray.append(corners2)

        # Draw and display the corners
        # drawChessboardCorners - Renders the detected chessboard corners.
        # The function draws individual chessboard corners detected either as red circles if
        # the board was not found, or as colored corners connected with lines if the board was found.
        img = cv.drawChessboardCorners(img, (rows, cols), corners2, ret)
        cv.imshow('Image',img)
        cv.waitKey(1)

# Destroying the image windows
cv.destroyAllWindows()

# Loading an image from the various images of chessboard to get the shape and to test the results of calibration by
# undistorting the image
I = cv.imread('Cam Calibrate Iphone\\GOPR001.jpg')
#plt.imshow(I), plt.show()
# Converting the image to grayscale
gray = cv.cvtColor(I, cv.COLOR_BGR2GRAY)

# Calibrate the camera and save the results
# calibrateCamera - Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.
# Output: mtx - Output 3x3 floating-point Camera Matrix
# dist - Output vector of distortion coefficients
# rvecs - Output vector of rotation vectors estimated for each pattern view
# tvecs - Output vector of translation vectors estimated for each pattern view
# The detailed description of these parameters is given in Report in Sec. 2 - Camera Calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objPntsArray, imgPntsArray, gray.shape[::-1], None, None)

print('The camera Matrix: ', mtx)
print('The distortion coefficients are : ', dist)

# Print the camera calibration error
error = 0
for i in range(len(objPntsArray)):
    imgP, _ = cv.projectPoints(objPntsArray[i], rvecs[i], tvecs[i], mtx, dist)
    error += cv.norm(imgPntsArray[i], imgP, cv.NORM_L2) / len(imgP)

print("Total error: ", error / len(objPntsArray))

# Extracting the shape of the image, Rows and Columns
h, w = I.shape[:2]

# getOptimalNewCameraMatrix - Returns the new camera matrix based on the free scaling parameter.
# The function computes and returns the optimal new camera matrix based on the free scaling parameter. By varying this
# parameter, you may retrieve only sensible pixels alpha=0 , keep all the original image pixels if there is valuable
# information in the corners alpha=1 , or get something in between.
newCamMat, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
print('Optimal Camera Matrix:', newCamMat)

# Undistorting the images
# undistort - Transforms an image to compensate for lens distortion by using the camera matrix, distortion coefficients
# and the optimal camera matrix computed in the previous steps
left_img_undist  = cv.undistort(I, mtx, dist, None, newCamMat)
cv.imwrite('Undistorted_1.bmp',left_img_undist)

print('RVECS: ', rvecs)
print('TVEC: ', tvecs)
np.savez('calibration.npz', mtx = mtx, dist = dist, rvecs = rvecs, tvecs = tvecs, newCamMat = newCamMat)
