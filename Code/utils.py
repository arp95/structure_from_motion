"""
 *  MIT License
 *
 *  Copyright (c) 2019 Arpit Aggarwal Shantam Bajpai
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the "Software"),
 *  to deal in the Software without restriction, including without
 *  limitation the rights to use, copy, modify, merge, publish, distribute,
 *  sublicense, and/or sell copies of the Software, and to permit persons to
 *  whom the Software is furnished to do so, subject to the following
 *  conditions:
 *
 *  The above copyright notice and this permission notice shall be included
 *  in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *  DEALINGS IN THE SOFTWARE.
"""


# header files loaded
import numpy as np
import cv2
import glob
from ReadCameraModel import *
from UndistortImage import *


# get the image
def get_image(file):
    """
    Inputs:
    
    file: the file to be read
    
    Outputs:
    
    image: numpy array representing the image
    """
    
    image = cv2.imread(file, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BayerGR2BGR)
    fx, fy, cx, cy, camera_image, LUT = ReadCameraModel("../data/model/")
    k_matrix = np.zeros((3, 3))
    k_matrix[0, 0] = fx
    k_matrix[1, 1] = fy
    k_matrix[2, 2] = 1
    k_matrix[0, 2] = cx
    k_matrix[1, 2] = cy
    image = UndistortImage(image, LUT)
    return (image, k_matrix)


#  get transformation matrices
def get_transformation_matrix(ptsLeft, ptsRight):
    
    # mean for ptsLeft and ptsRight
    ptsLeft_mean_x = np.mean(ptsLeft[:, 0])
    ptsLeft_mean_y = np.mean(ptsLeft[:, 1])
    ptsRight_mean_x = np.mean(ptsRight[:, 0])
    ptsRight_mean_y = np.mean(ptsRight[:, 1])
    
    # get updated ptsLeft and ptsRight
    sum_ptsLeft = 0.0
    sum_ptsRight = 0.0
    for index in range(0, len(ptsLeft)):
        ptsLeft[index][0] = ptsLeft[index][0] - ptsLeft_mean_x
        ptsLeft[index][1] = ptsLeft[index][1] - ptsLeft_mean_y
        sum_ptsLeft = sum_ptsLeft + ((ptsLeft[index][0] ** 2) + (ptsLeft[index][1] ** 2))
        
    for index in range(0, len(ptsRight)):
        ptsRight[index][0] = ptsRight[index][0] - ptsRight_mean_x
        ptsRight[index][1] = ptsRight[index][1] - ptsRight_mean_y
        sum_ptsRight = sum_ptsRight + ((ptsRight[index][0] ** 2) + (ptsRight[index][1] ** 2))
    sum_ptsLeft = sum_ptsLeft / len(ptsLeft)
    sum_ptsRight = sum_ptsRight / len(ptsRight)
    
    # scale factor for ptsLeft and ptsRight
    scale_ptsLeft = np.sqrt(2) / np.sqrt(sum_ptsLeft)
    scale_ptsRight = np.sqrt(2) / np.sqrt(sum_ptsRight)
    
    # get transformation matrices
    ptsLeft_transformation_matrix = np.dot(np.array([[scale_ptsLeft, 0, 0], [0, scale_ptsLeft, 0], [0, 0, 1]]), np.array([[1, 0, -ptsLeft_mean_x], [0, 1, -ptsLeft_mean_y], [0, 0, 1]]))
    ptsRight_transformation_matrix = np.dot(np.array([[scale_ptsRight, 0, 0], [0, scale_ptsRight, 0], [0, 0, 1]]), np.array([[1, 0, -ptsRight_mean_x], [0, 1, -ptsRight_mean_y], [0, 0, 1]]))
    
    # get normalized points
    for index in range(0, len(ptsLeft)):
        ptsLeft[index][0] = ptsLeft[index][0] * scale_ptsLeft
        ptsLeft[index][1] = ptsLeft[index][1] * scale_ptsLeft
    
    for index in range(0, len(ptsRight)):
        ptsRight[index][0] = ptsRight[index][0] * scale_ptsRight
        ptsRight[index][1] = ptsRight[index][1] * scale_ptsRight
    
    # return matrices
    return (ptsLeft, ptsRight, ptsLeft_transformation_matrix, ptsRight_transformation_matrix)


# get keypoints between frame 1 and frame 2
def get_keypoints(image1, image2):
    """
    Inputs:
    
    image1: left image
    image2: right image
    
    Outputs:
    
    ptsLeft: point correspondences for left image
    ptsRight: point correspondences for right image
    """
    
    # use sift keypoint to get the points
    sift = cv2.ORB_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)
    des1 = np.asarray(des1, np.float32)
    des2 = np.asarray(des2, np.float32)
    flann = cv2.FlannBasedMatcher(dict(algorithm = 0, trees = 5), dict(checks = 50))
    matches = flann.knnMatch(des1, des2, k = 2)
    good = []
    ptsLeft = []
    ptsRight = []

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            ptsRight.append(kp2[m.trainIdx].pt)
            ptsLeft.append(kp1[m.queryIdx].pt)

    ptsLeft = np.int32(ptsLeft)
    ptsRight = np.int32(ptsRight)
    return (ptsLeft, ptsRight)


# get fundamental matrix with ransac
def get_fundamental_matrix_ransac(ptsLeft, ptsRight):
    """
    Inputs:
    
    ptsLeft: array of 8 points for left image
    ptsRight: array of 8 points for right image
    
    Outputs:
    
    fundamental_mat: fundamental matrix of size (3 x 3)
    """
    
    # normalise points
    (ptsLeft, ptsRight, ptsLeft_transformation_matrix, ptsRight_transformation_matrix) = get_transformation_matrix(ptsLeft, ptsRight)

    # ransac for better matrix estimation
    iterations = 500
    error = 1000000
    best_fundamental_matrix = get_fundamental_matrix(ptsLeft[:8], ptsRight[:8], ptsLeft_transformation_matrix, ptsRight_transformation_matrix)
    for iteration in range(0, iterations):
        
        indexes = np.random.choice(np.array(ptsLeft).shape[0], 8, replace=False)
        selected_ptsLeft = []
        selected_ptsRight = []
        not_selected_ptsLeft = []
        not_selected_ptsRight = []
        for index in range(0, len(ptsLeft)):
            flag = 0
            for k in range(0, len(indexes)):
                if(index == indexes[k]):
                    flag = 1
                    break
            
            if(flag):
                selected_ptsLeft.append(ptsLeft[index])
                selected_ptsRight.append(ptsRight[index])
            else:
                not_selected_ptsLeft.append(ptsLeft[index])
                not_selected_ptsRight.append(ptsRight[index])
    
        estimated_fundamental_mat = get_fundamental_matrix(selected_ptsLeft, selected_ptsRight, ptsLeft_transformation_matrix, ptsRight_transformation_matrix)
        estimated_error = 0.0
        for index in range(0, len(not_selected_ptsLeft)):
            x_right = np.array([[not_selected_ptsRight[index][0]], [not_selected_ptsRight[index][1]], [1]])
            x_left = np.array([[not_selected_ptsLeft[index][0]], [not_selected_ptsLeft[index][1]], [1]])
            value1 = np.dot(x_right.T, np.dot(estimated_fundamental_mat, x_left))
            value2 = np.dot(x_left.T, np.dot(estimated_fundamental_mat, x_right))
            estimated_error = estimated_error + ((value1 * value1) + (value2 * value2))
        if(estimated_error < error):
            error = estimated_error
            best_fundamental_matrix = estimated_fundamental_mat
    
    # return fundamental matrix
    return best_fundamental_matrix
    
    
# get fundamental matrix
def get_fundamental_matrix(pointsLeft, pointsRight, ptsLeft_transformation_matrix, ptsRight_transformation_matrix):
    """
    Inputs:
    
    pointsLeft: array of 8 points for left image
    pointsRight: array of 8 points for right image
    ptsLeft_transformation_matrix: transformation matrix for left image
    ptsRight_transformation_matrix: transformation matrix for right image
    
    Outputs:
    
    fundamental_mat: fundamental matrix of size (3 x 3)
    """
    
    # construct a matrix
    a_matrix = []
    for index in range(0, len(pointsLeft)):
        x, y = pointsLeft[index][0], pointsLeft[index][1]
        u, v = pointsRight[index][0], pointsRight[index][1]
        a_matrix.append([x * u, y * u, u, x * v, y * v, v, x, y, 1])
    
    # svd of A
    a_matrix = np.array(a_matrix)
    u, s, vh = np.linalg.svd(a_matrix)
        
    # compute fundamental_mat
    vh = vh.T
    fundamental_mat = vh[:, -1]
    fundamental_mat = fundamental_mat.reshape((3, 3))
    
    # enforce rank 2 constraint and update fundamental_mat
    u, s, vh = np.linalg.svd(fundamental_mat)
    s[2] = 0
    fundamental_mat = np.dot(u, np.dot(np.diag(s), vh))
    
    # un-normalize fundamental_mat
    fundamental_mat = np.dot(ptsRight_transformation_matrix.T, np.dot(fundamental_mat, ptsLeft_transformation_matrix))
    fundamental_mat = fundamental_mat / fundamental_mat[2, 2]
    
    # return the matrix
    return fundamental_mat


# estimate essential matrix
def get_essential_matrix(fundamental_matrix, k_matrix):
    """
    Inputs:
    
    fundamental_matrix: Matrix that relates image coordinates in one image to the other
    k_matrix: the calibration matrix of the camera
    
    Outputs:
    
    essential_matrix: return essential matrix
    """
    
    essential_matrix = np.dot(k_matrix.T, np.dot(fundamental_matrix, k_matrix))
    u, s, vh = np.linalg.svd(essential_matrix)
    s[0] = 1
    s[1] = 1
    s[2] = 0
    essential_matrix = np.dot(u, np.dot(s, vh))
    essential_matrix = essential_matrix / np.linalg.norm(essential_matrix)
    
    # return matrix
    return essential_matrix
