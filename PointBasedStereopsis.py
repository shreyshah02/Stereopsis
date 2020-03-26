import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# This code is derived from the Matlab code provided by Dr. Peters in the Stereopsis lecture slides

# pil - point in left image, pir - point in right image
def Depth_Estimation(pil, Rl, Tl, pir,Rr, Tr):
    # Rotating the image points into alignment with the world coordinates
    # pwl = np.matmul(Rl, pil)
    # pwr = np.matmul(Rr, pir)
    #
    # # Construct a covariance matrix between pwl and pwr
    # # Construct a Covariance Matrix between pwl and pwr
    # # Magnitude squared of the left point
    # pl2 = np.matmul(np.transpose(pwl), pwl)
    # # Magnitude squared of the right point
    # pr2 = np.matmul(np.transpose(pwr), pwr)
    # # Covariance of the right point with the right
    # plr = np.matmul(np.transpose(pwl), pwr)
    # # Covariance of the left point with the right
    # prl = np.matmul(np.transpose(pwr), pwl)
    #
    # C = np.array(([pl2, plr],[prl, pr2]))
    #
    # # Computing the Inverse Matrix
    # C_inv = np.linalg.inv(C)
    # #print('Shape of C_inv: ',C_inv.shape)
    #
    # # Compute the Baseline :
    # base = Tr - Tl
    #
    # # Project each point onto the baseline
    # b1 = np.matmul(np.transpose(base), pwl)
    # b2 = np.matmul(np.transpose(base), pwr)
    #
    # b = np.array(([b1],[b2]))
    # #print('shape of b: ',b.shape,' value: ', b)
    # # Transform the baseline projections into the pwl - pwr cdt sys
    # lmbda = np.matmul(C_inv, b)
    # #print('Shape of Lambda: ', lmbda.shape)
    #
    # # re express each in terms of world coordinates
    # pwlest = np.matmul(pwl,lmbda[0,0]) + Tl
    # pwres = np.matmul(pwr,lmbda[1,0]) + Tr
    #
    # # Take their Average
    # pw = (pwlest + pwres)/2
    # return 154.2*pw

    # Rotating the image pointsinto alignment with the world coordinates
    pwl = np.matmul(Rl, pil)
    pwr = np.matmul(Rr, pir)
    #print('Shape of pwl: ', pwl.shape)

    # Construct a Covariance Matrix between pwl and pwr
    # Magnitude squared of the left point
    pl2 = np.matmul(np.transpose(pwl), pwl)
    pl2 = pl2[0,0]
    # Magnitude squared of the right point
    pr2 = np.matmul(np.transpose(pwr), pwr)
    pr2 = pr2[0,0]
    # Covariance of the right point with the right
    plr = np.matmul(np.transpose(pwl), pwr)
    plr = plr[0,0]
    # Covariance of the left point with the right
    prl = np.matmul(np.transpose(pwr), pwl)
    prl = prl[0,0]

    C = np.array(([pl2, plr],[prl, pr2]))

    # Computing the Inverse Matrix
    C_inv = np.linalg.inv(C)
    #print('Shape of C_inv: ',C_inv.shape)

    # Compute the Baseline :
    base = Tr - Tl
    # Multiplying with estimated translation of 6inch = 152.4 mm
    #base = 152.4*base

    # Project each point onto the baseline
    b1 = np.matmul(np.transpose(base), pwl)
    b2 = np.matmul(np.transpose(base), pwr)
    b1 = b1[0,0]
    b2 = b2[0,0]
    #print(b1, b1.shape)

    b = np.array((b1,b2))
    #print('shape of b: ',b.shape,' value: ', b)
    b = np.expand_dims(b, axis = 1)
    #print('shape of b: ', b.shape, ' value: ', b)
    # Transform the baseline projections into the pwl - pwr cdt sys
    lmbda = np.matmul(C_inv, b)
    #print('Shape of Lambda: ', lmbda.shape)
    l1 = lmbda[0,0]
    l2 = lmbda[1,0]

    # re express each in terms of world coordinates
    pwlest = pwl * l1 + Tl
    pwres = pwr * l2 + Tr

    # Take their Average
    pw = (pwlest + pwres)/2
    return 152.4*pw

#cv.namedWindow('Image', cv.WINDOW_GUI_NORMAL)

I_left = cv.imread('laptop_left.jpg')
I_right = cv.imread('laptop_right.jpg')

#cv.imshow('Image', I_left)
#cv.waitKey(0)

#plt.imshow(I_left), plt.show()
Rl = np.identity(3)
Tl = np.zeros((3,1))

# R_and_T = np.load('Rot_and_Trns.npz')
#
# Rr = R_and_T['Rot']
# Tr = R_and_T['Trans']

# Loading the Rotational matrix and translation vector computed during the epeipolar stereopsis
R_and_T = np.load('Algebraic_Rot_T.npz')
Rr = R_and_T['R']
Tr = R_and_T['t']
# Loading the Conjugate Point Pairs
Point_Pairs = np.load('Conjugate_Point_Pairs.npz')
pts_left = Point_Pairs['Left']
pts_right = Point_Pairs['Right']
print('Points in Left:\n',pts_left)
print('Points in Right:\n', pts_right)
#print('TestRun:', pts_left[1][:], pts_left[1,:])

# Initializing the depth variable to store the depths of images
depth = []
# creating 3 x 1 zero vectors for holding the coordinates of the corresponding points from the left and right image respectively
left = np.zeros((3,1))
right = np.zeros((3,1))

for i in range(0,4):
    # Storing the coordinates in left and right vectors and converting it to homogeneous
    left[:2,0] = pts_left[i,:]
    left[2,0]=1
    #print(left)
    # print(left)
    right[:2,0] = pts_right[i,:]
    right[2,0]= 1
    # Calling the function to generate the world coordinates for the corresponding conjugate points
    p_world = Depth_Estimation(left, Rl, Tl, right, Rr, Tr)
    print('World Point ', i, 'th point = ', p_world)
    # Appending the z-coordinate as depth
    depth.append(p_world[2,0])

print('List of all Depths= \n', depth)

