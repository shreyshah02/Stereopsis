import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Loading the camera parameters calculated during camera calibration
cam_parameters = np.load('calibration.npz')
# Camera Matrix and the optimal Camera Matrix
k = cam_parameters['mtx']
newK = cam_parameters['newCamMat']
# print('The camera Matrix: \n', k)
# Distortion Coefficients
dist = cam_parameters['dist']

# Loading the Stereo Image Pair
left_img = cv.imread('laptop_left.jpg')
right_img = cv.imread('laptop_right.jpg')

# Extracting the shape of the image
R, C, B = left_img.shape

# Converting the images to grayscale
gray_L = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
gray_R = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)

# Using SIFT for matching the Keypoints
# Create object of SIFT
sift = cv.xfeatures2d.SIFT_create()

# Detect the keypoints
kp_L, des_L = sift.detectAndCompute(gray_L, None)
kp_R, des_R = sift.detectAndCompute(gray_R, None)

# Using FLANN based Matcher
# FLANN stands for Fast Library for Approximate Nearest Neighbors.

# The Detailed Description is given in the report

# FLANN Parameters
FLANN_INDEX_KDTREE = 1

# For FLANN based matcher, we need to pass two dictionaries which specifies the algorithm to be used, its related
# parameters etc. First one is IndexParams.
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
# Second dictionary is the SearchParams. It specifies the number of times the trees in the index should be recursively
# traversed. Higher values gives better precision, but also takes more time.
search_params = dict(checks=50)

# Creating the Object of FlannBased Matcher
flann = cv.FlannBasedMatcher(index_params, search_params)
# Performing the Matching between the Stereo images using knnMatch
# knnMatch - Finds the k best matches for each descriptor from a query set
# k=2 so that we can apply ratio test explained by D.Lowe in his paper.
matches = flann.knnMatch(des_L, des_R,k=2)

# Generating the List of Best matches and the keypoints in both Images Corresponding to those Matches
good = []
matched_L = []
matched_R = []

# # ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        # Appending The good match
        good.append(m)
        # Extracting and appending the coordinates of the keypoint corresponding to good match in left image
        matched_L.append(kp_L[m.queryIdx].pt)
        # Extracting and appending the coordinates of the keypoint corresponding to good match in right image
        matched_R.append(kp_R[m.trainIdx].pt)

# Converting the List of coordinates of keypoints in both images corresponding to
# good matches to type numpy array
matched_L = np.array(matched_L)
matched_R = np.array(matched_R)
print('Shape of matched keypoints: ', matched_L.shape)

# Defining the Criteria to be applied while undistorting the keypoint coordinates from both images
# using cv2.undistortPointsIter
# 30 - specifies the maximum no. of iterations
# 1e-6 - specifies the required accuracy
# The Iteration stops when either 30 iterations have been performed or the required accuracy is reached
# while undistorting the keypoints
# Criteria
crit_cal = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

# Changing the shape of the Matched Keypoint Coordinates from N x 2 to N x 1 X 2
# As the function undistortPointsIter requires keypoint in this shape
# Adding New axis to the matched points using numpy.newaxis
matched_L_na = matched_L[:, np.newaxis, :]
matched_R_na = matched_R[:, np.newaxis, :]

# Undistorting the points using the function cv2.undistortPointsIter
# This Function Computes the ideal point coordinates from the observed point coordinates iteratively
# Until either the desired accuracy is achieved or the maximum number of iterations is achieved

# Detailed explanation is in the report in section :

undist_pts_L = cv.undistortPointsIter(matched_L_na, k, dist, R = None, P = newK, criteria=crit_cal)
undist_pts_R = cv.undistortPointsIter(matched_R_na, k, dist, R = None, P = newK, criteria=crit_cal)
print('Shape of the returned Undistorted points: ', undist_pts_L.shape)

# Now we the undistorted points corresponding to the Best matches between the sterio pair of images
# Now we will compute the Fundamental Matrix
#

# Computing the Fundamental Matrix
#F, mask = cv.findFundamentalMat(undist_pts_L, undist_pts_R, cv.FM_RANSAC, ransacReprojThreshold=0.1, confidence=0.99)
# Fundamental Matrix is a 3 x 3 Matrix which transforms a point in one image to a corresponding line in the second image
# Such Lines are called epipolar Lines, where these images form a stereo pair

# Detailed Explanation is given in the report in Section

# We use the function findFundamentalMat to compute the fundamental matrix
# The function calculates the fundamental matrix using Ransac and returns the found fundamental matrix
# The detailed description of this function is given in the report under section
# F - Fundamental Matrix
# mask - mask is used to extract the Inliner points from the matched points
# Inliner points - matched keypoints which are compliant with the epipolar geometry
F, mask = cv.findFundamentalMat(undist_pts_L, undist_pts_R, cv.FM_RANSAC, ransacReprojThreshold=1, confidence=0.99)
print('Fundamental Matrix: \n', F)
# saving the FUndamental Matrix
np.savez('Fundamental_Matrix.npz', F= F)

# Computing Inliner Points
# Inliner points - matched keypoints which are compliant with the epipolar geometry
inLeft = matched_L[mask.ravel() == 1]
inRight = matched_R[mask.ravel() == 1]
npts = inLeft.shape[0]

# The following code is taken from:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
# This Function draws the epilines computed
def draw_epilines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape[:2]
    for r,pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(np.int32(pt1)), 5, color, -1)
        img2 = cv.circle(img2, tuple(np.int32(pt2)), 5, color, -1)
    return img1, img2


# Undistorting the original images
# undistort - Transforms an image to compensate for lens distortion by using the camera matrix, distortion coefficients
# and the optimal camera matrix computed in the previous steps
left_img_undist  = cv.undistort(left_img, k, dist, None, newK)
right_img_undist = cv.undistort(right_img, k, dist, None, newK)
cv.imwrite('Undist_Laptop_L.bmp', left_img_undist)
cv.imwrite('Undist_Laptop_R.bmp', right_img_undist)

# Find epilines corresponding to points in right image (second image)
# drawing its lines on the left image
lines_L = cv.computeCorrespondEpilines(inRight.reshape(-1, 1, 2), 2, F)
lines_L = lines_L.reshape(-1, 3)
#left_img_lines, img6 = draw_epilines(gray_L, gray_R, lines_L, inLeft, inRight)
left_img_lines, img6 = draw_epilines(left_img_undist.copy(), right_img_undist.copy(), lines_L, inLeft, inRight)

# Finding the epilines corresponding to points in left image (First Image) and
# drawing its lines on the right image

lines_R = cv.computeCorrespondEpilines(inLeft.reshape(-1, 1, 2), 1, F)
lines_R = lines_R.reshape(-1, 3)
#right_img_lines, img4 = draw_epilines(gray_R, gray_L, lines_R, inRight, inLeft)
right_img_lines, img4 = draw_epilines(right_img_undist.copy(), left_img_undist.copy(), lines_R, inRight, inLeft)

cv.imwrite('Epilines_laptop_L.bmp', left_img_lines)
cv.imwrite('Epilines_laptop_R.bmp', right_img_lines)

new_right = inRight.reshape(-1, 1, 2)
new_left = inLeft.reshape(-1, 1, 2)
np.savez('Inliners.npz', inLeft = new_left, inRight = new_right, npts = npts)

# stereoRectifyuncalibrated(). The function computes the rectification transformations without knowing intrinsic
# parameters of the cameras and their relative position in the space.
# This method takes the following input parameter – 1. Points1 – array of key features i.e. the inlier points in
# image1, 2. Points – array of key features i.e. the inlier points in image2, 3. F – the fundamental matrix,
# 4. imgSize – the size of the image as a tuple of columns, rows. The method, using Hartley’s algorithm, returns the
# left and right homography matrices.
ret, Hl, Hr = cv.stereoRectifyUncalibrated(new_left[:, :, 0:2], new_right[:, :, 0:2], F, (C, R))

Hl/=Hl[2,2]
Hr/=Hr[2,2]

# Printing the Homographies
print('Left Homography: \n', Hl)
print('The Right Homography: \n',Hr)

# Rectifying the Images:
# Doubt Between R,C and C, R
rect_left = cv.warpPerspective(left_img_undist, Hl, (C, R))
rect_right = cv.warpPerspective(right_img_undist, Hr, (C,R))

cv.imwrite('Rect_Left_laptop.bmp', rect_left)
cv.imwrite('Rect_right_laptop.bmp', rect_right)

# Computing the disparity map of the images
# Setting the Disparity parameters

win_size = 5
min_disp = 0
max_disp = 64
num_disp = max_disp - min_disp

# Creating the Block Matching Object
stereo = cv.StereoSGBM_create(minDisparity=min_disp, blockSize=5, uniquenessRatio=5, speckleWindowSize=5, speckleRange=5, disp12MaxDiff=1, P1 =8*3*win_size**2,
                              P2=32*3*win_size**2)
disparity_map = stereo.compute(rect_left, rect_right)
# The Following code is taken from the site:
# https://shkspr.mobi/blog/2018/04/reconstructing-3d-models-from-the-last-jedi/

# Creating the object of RightMatcher with stereo as parameter
right_matcher = cv.ximgproc.createRightMatcher(stereo)

# Creating the object of WLS Filter with stereo object sent as parameter
wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
# Setting the Lambda value = 800000 for the wls filter and setting the sigma = 1.2
wls_filter.setLambda(80000)
wls_filter.setSigmaColor(1.2)

# Computing the disparity using StereoSGBM between right and left rectified image
disparity_left = stereo.compute(rect_right, rect_left)
# Computing the disparity map between left and right disparity image using rightmatcher
disparity_right = right_matcher.compute(rect_left, rect_right)
# Converting the type to np.int16 for both the disparities
disparity_left = np.int16(disparity_left)
disparity_right = np.int16(disparity_right)

# Using the wls filter with disparity_left, rect_left and disparity_right as parameters
# in order to get a much refined disparity map by using the output to normalize the disparity map
filteredImg = wls_filter.filter(disparity_left, rect_left, None, disparity_right)

# Normalizing the Disparity map
depth_map = cv.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
depth_map = np.uint8(depth_map)
#depth_map = cv.bitwise_not(depth_map) # Invert image. Optional depending on stereo pair
imcolor= cv.applyColorMap(depth_map, cv.COLORMAP_JET)
plt.imsave("depth.png", imcolor)


#disparity_map = cv.normalize(disparity_map, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
#Image = cv.applyColorMap(disparity_map, cv.COLORMAP_JET)
#plt.imshow(disparity_map, 'gray'), plt.show()
# cv.imwrite('Disparity_Map.bmp', disparity_map)






