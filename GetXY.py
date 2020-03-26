import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

a = np.array([0,0], dtype = 'float32')

# Function to get the coordinates of the point in the image selected by the mouse
def getxy(event, x, y, flags, param):
    global a
    if event == cv.EVENT_LBUTTONDOWN:
        # Appending the extracted coordinates to a
        a= np.vstack([a, np.hstack([x,y])])

# Loading the images
I_left = cv.imread('laptop_left.jpg')
I_right = cv.imread('laptop_right.jpg')
# Extracting the shape of the image
R, C, B = I_left.shape
I_left = cv.resize(I_left,(np.int32(R/5), np.int32(C/5)))
I_right = cv.resize(I_right,(np.int32(R/5), np.int32(C/5)))
cv.namedWindow('Image', cv.WINDOW_AUTOSIZE)
# setting the mousecallback on the window Image
cv.setMouseCallback('Image', getxy)

#show the image
# Displaying the left image and extracting its coordinates
cv.imshow('Image', I_left)
cv.waitKey(0)
# Extracting the first 4 values from a as they were selected from the left image
Left = a[1:,:]
print('Left: \n',Left)
print(type(Left))
# Displaying the right image and extracting its coordinates
cv.imshow('Image', I_right)
cv.waitKey(0)
# Extracting the remaining 4 values from a as they were selected from the right image
Right = a[5:,:]
print('Right: \n',Right)
# Saving the coordinates from the left image and right image in order to use it while performing point stereopsis
np.savez('Conjugate_Point_Pairs.npz', Left = Left, Right = Right)
cv.destroyAllWindows()


