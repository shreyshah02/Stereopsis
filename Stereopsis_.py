import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Loading the images
left_img = cv.imread('laptop_left.jpg')
right_image = cv.imread('laptop_right.jpg')

# Converting the images to grayscale images
imgL_gray = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
imgR_gray = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)

# Creating the object of StereoBM
stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
# Calculating the disparity
disparity = stereo.compute(imgL_gray, imgR_gray)
# Saving the image
cv.imwrite('Disp_SBM_Normal.bmp',disparity)

# Displaying the Disparity Image
plt.imshow(disparity, 'gray'), plt.title('Disparity'), plt.show()

# Blurring the images
imgL_blur = cv.GaussianBlur(imgL_gray, (5,5), 0)
imgR_blur = cv.GaussianBlur(imgR_gray, (5,5), 0)

# Computing the disparity image after blurring
disparity_blur = stereo.compute(imgL_blur, imgR_blur)

cv.imwrite('Disp_SBM_Blured.bmp',disparity_blur)

# Displaying the Disparity image after blurring
plt.imshow(disparity_blur,'gray'), plt.title('Disparity after Gaussian Blurring'), plt.show()

# Computing the gradient images using the Sobel operator
imgL_sobel = cv.Sobel(imgL_gray,-1, 1, 1, ksize=5)
imgR_sobel = cv.Sobel(imgR_gray, -1, 1,1, ksize=5)

# Computing the disparity image on gradient images
disparity_grdnt = stereo.compute(imgL_sobel, imgR_sobel)

cv.imwrite('Disp_SBM_Gradient.bmp',disparity_grdnt)

# Displaying the disparity image of gradient images
plt.imshow(disparity_grdnt,'gray'), plt.title('Disparity on Gradient Images'), plt.show()


