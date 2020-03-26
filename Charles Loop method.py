import epipolar
import numpy as np
import numpy.linalg as la
import pylab as pl
import cv2 as cv



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

F = np.array([[ 8.61943857e-09, -6.19474251e-07  ,8.14433596e-04],
[ 5.07556996e-07  ,7.50609505e-08 ,-1.25413593e-02],
[-8.55189606e-04 , 1.20364617e-02 , 1.00000000e+00]])

K = np.array([[3.37289677e+03, 0.00000000e+00, 1.59491773e+03],
 [0.00000000e+00 ,3.37219712e+03, 1.86830009e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

d = np.array([[-0.03978204,  0.01997505, -0.00228564, -0.00026135,  0.02453989]])

## ---- Rectification based on found fundamental matrix
def rectify_images(img1, x1, img2, x2, K, d, F, shearing=False):
    imsize = (img1.shape[1], img1.shape[0])
    H1, H2 = epipolar.rectify_uncalibrated(x1, x2, F, imsize)


    if shearing:
        S = epipolar.rectify_shearing(H1, H2, imsize)
        H1 = S.dot(H1)
    print('The homographies are: \n', H1, '\nR\n', H2)
    rH = la.inv(K).dot(H1).dot(K)
    lH = la.inv(K).dot(H2).dot(K)
    print('The homographies are: \n',lH, '\nR\n',rH)

    map1x, map1y = cv.initUndistortRectifyMap(K, d, rH, K, imsize,
                                               cv.CV_16SC2)
    map2x, map2y = cv.initUndistortRectifyMap(K, d, lH, K, imsize,
                                               cv.CV_16SC2)

    # Convert the images to RGBA (add an axis with 4 values)
    img1 = np.tile(img1[:,:,np.newaxis], [1,1,4])
    img1[:,:,3] = 255
    img2 = np.tile(img2[:,:,np.newaxis], [1,1,4])
    img2[:,:,3] = 255

    rimg1 = cv.remap(img1, map1x, map1y,
                      interpolation=cv.CV_INTER_LINEAR,
                      borderMode=cv.BORDER_CONSTANT,
                      borderValue=(0,0,0,0))
    rimg2 = cv.remap(img2, map2x, map2y,
                      interpolation=cv.CV_INTER_LINEAR,
                      borderMode=cv.BORDER_CONSTANT,
                      borderValue=(0,0,0,0))

    # Put a red background on the invalid values
    # TODO: Return a mask for valid/invalid values
    # TODO: There is aliasing hapenning on the images border. We should
    # invalidate a margin around the border so we're sure we have only valid
    # pixels
    rimg1[rimg1[:,:,3] == 0,:] = (255,0,0,255)
    rimg2[rimg2[:,:,3] == 0,:] = (255,0,0,255)

    return rimg1, rimg2


pl.figure()
pl.suptitle('With shearing transform')
rimg1, rimg2 = rectify_images(left_img, matched_L, right_img, matched_R, K, d, F, shearing=True)
epipolar.show_rectified_images(rimg1, rimg2)


pl.show()

#Source: https://github.com/julienr/cvscripts/tree/master/rectification