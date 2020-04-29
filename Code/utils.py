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
        x = ptsLeft[index][0] - ptsLeft_mean_x
        y = ptsLeft[index][1] - ptsLeft_mean_y
        sum_ptsLeft = sum_ptsLeft + np.sqrt((x ** 2) + (y ** 2))
        
    for index in range(0, len(ptsRight)):
        x = ptsRight[index][0] - ptsRight_mean_x
        y = ptsRight[index][1] - ptsRight_mean_y
        sum_ptsRight = sum_ptsRight + np.sqrt((x ** 2) + (y ** 2))
    sum_ptsLeft = sum_ptsLeft / len(ptsLeft)
    sum_ptsRight = sum_ptsRight / len(ptsRight)
    
    # scale factor for ptsLeft and ptsRight
    scale_ptsLeft = np.sqrt(2) / sum_ptsLeft
    scale_ptsRight = np.sqrt(2) / sum_ptsRight
    
    # get transformation matrices
    ptsLeft_transformation_matrix = np.dot(np.array([[scale_ptsLeft, 0, 0], [0, scale_ptsLeft, 0], [0, 0, 1]]), np.array([[1, 0, -ptsLeft_mean_x], [0, 1, -ptsLeft_mean_y], [0, 0, 1]]))
    ptsRight_transformation_matrix = np.dot(np.array([[scale_ptsRight, 0, 0], [0, scale_ptsRight, 0], [0, 0, 1]]), np.array([[1, 0, -ptsRight_mean_x], [0, 1, -ptsRight_mean_y], [0, 0, 1]]))
    
    # get normalized points
    for index in range(0, len(ptsLeft)):
        point = np.array([[ptsLeft[index][0]], [ptsLeft[index][1]], [1]])
        point = np.dot(ptsLeft_transformation_matrix, point)
        ptsLeft[index] = np.array([point[0][0] / point[2][0], point[1][0] / point[2][0]])
    
    for index in range(0, len(ptsRight)):
        point = np.array([[ptsRight[index][0]], [ptsRight[index][1]], [1]])
        point = np.dot(ptsRight_transformation_matrix, point)
        ptsRight[index] = np.array([point[0][0] / point[2][0], point[1][0] / point[2][0]])
    
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
        if m.distance < (0.8 * n.distance):
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
    iterations = 5000
    threshold = 0.03
    count = 0
    best_fundamental_matrix = get_fundamental_matrix(ptsLeft[:8], ptsRight[:8], ptsLeft_transformation_matrix, ptsRight_transformation_matrix)
    for iteration in range(0, iterations):
        
        indexes = np.random.choice(np.array(ptsLeft).shape[0], 8, replace=False)
        selected_ptsLeft = []
        selected_ptsRight = []
        for index in range(0, len(ptsLeft)):
            flag = 0
            for k in range(0, len(indexes)):
                if(index == indexes[k]):
                    flag = 1
                    break
            
            if(flag):
                selected_ptsLeft.append(ptsLeft[index])
                selected_ptsRight.append(ptsRight[index])
    
        estimated_fundamental_mat = get_fundamental_matrix(selected_ptsLeft, selected_ptsRight, ptsLeft_transformation_matrix, ptsRight_transformation_matrix)
        estimated_count = 0
        for index in range(0, len(ptsLeft)):
            x_right = np.array([[ptsRight[index][0]], [ptsRight[index][1]], [1]])
            x_left = np.array([[ptsLeft[index][0]], [ptsLeft[index][1]], [1]])
            error = np.dot(x_right.T, np.dot(estimated_fundamental_mat, x_left))
            
            if(np.abs(error) < threshold):
                estimated_count = estimated_count + 1
                
        if(estimated_count > count):
            count = estimated_count
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
    essential_matrix = np.dot(u, np.dot(np.diag(s), vh))
    essential_matrix = essential_matrix / np.linalg.norm(essential_matrix)
    
    # return matrix
    return essential_matrix


#function to extract camera poses from essential matrix
def get_camera_poses(essential_matrix):
    """
    Inputs:
    
    essential_matrix: return essential matrix
    
    Outputs:
    
    (r1, r2, r3, r4, c1, c2, c3, c4): four possible camera poses, that is, four rotation matrices and four translation matrices
    """
    
    # define rotation matrix and get svd decomposition of essential matrix
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    u, d, v = np.linalg.svd(essential_matrix)

    # define four camera poses (c1, r1), (c2, r2), (c3, r3), (c4, r4)
    c1 = u[:, 2]
    c2 = -u[:, 2]
    c3 = u[:, 2]
    c4 = -u[:, 2]
    r1 = np.dot(u, np.dot(w, v))
    r2 = np.dot(u, np.dot(w, v))
    r3 = np.dot(u, np.dot(w.T, v))
    r4 = np.dot(u, np.dot(w.T, v))

    if np.linalg.det(r1) < 0:
        r1 = -r1
        c1 = -c1
    if np.linalg.det(r2) < 0:
        r2 = -r2
        c2 = -c2
    if np.linalg.det(r3) < 0:
        r3 = -r3
        c3 = -c3
    if np.linalg.det(r4) < 0:
        r4 = -r4
        c4 = -c4
    
    # reshape the translation matrices
    c1 = c1.reshape(-1, 1)
    c2 = c2.reshape(-1, 1)
    c3 = c3.reshape(-1, 1)
    c4 = c4.reshape(-1, 1)
    
    # return four possible camera poses
    return [[np.array(c1), np.array(c2), np.array(c3), np.array(c4)], [np.array(r1), np.array(r2), np.array(r3), np.array(r4)]]


# determines whether the point is in front of camera or not
def is_point_in_front(camera_pose, point):
    """
    Inputs:
    
    camera_pose: the camera pose
    point: the 3D point in camera coordinate system
    
    Output: 
    
    True/False: tells whether the point is in front of the camera or not
    """
    
    r = camera_pose[:, :-1]
    t = camera_pose[:, -1:]

    # cheirality condition
    if((r[2, :] * (point + r.T * t)) > 0):
        return True
    return False 


# performs linear triangulation
def get_linear_triangulation(camera_pose_1, camera_pose_2, pointLeft, pointRight, k_matrix):
    """
    Inputs:
    
    camera_pose_1: the base camera pose
    camera_pose_2: the camera pose
    pointLeft: the image point in the left image
    pointRight: the image point in the right image
    k_matrix: the camera matrix
    
    Output: 
    
    point: the 3D point in camera coordinate system
    """

    # get the cross-product matrix for point-1 and point-2
    pointLeft_cross_product = np.array([[0, -1, pointLeft[1]], [1, 0, -pointLeft[0]], [-pointLeft[1], pointLeft[0], 0]])
    pointRight_cross_product = np.array([[0, -1, pointRight[1]], [1, 0, -pointRight[0]], [-pointRight[1], pointRight[0], 0]])
    
    # get the m_matrix
    camera_pose_1 = camera_pose_1[:-1, :]
    m_matrix = np.vstack([np.dot(pointLeft_cross_product, np.dot(k_matrix, camera_pose_1)), np.dot(pointRight_cross_product, np.dot(k_matrix, camera_pose_2))])
    a_matrix = m_matrix[:, :-1]
    b = -m_matrix[:, -1:]
    
    # get the 3D point
    point = np.dot(np.dot(np.linalg.inv(np.dot(a_matrix.T, a_matrix)), a_matrix.T), b)
    
    # return point
    return point
    
    
# estimate the best camera pose
def get_best_camera_pose(translation_matrices, rotation_matrices, base_pose, ptsLeft, ptsRight, k_matrix):
    """
    Inputs:
    
    translation_matrices: set of translation matrices
    rotation_matrices: set of rotation matrices
    base_pose: the base pose
    ptsLeft: the point correspondences for left image
    ptsRight: the point correspondences for right image
    
    Output: 
    
    best_pose: the best camera pose for the frame
    """
    
    # form four possible camera matrices
    camera_pose_1 = np.hstack([rotation_matrices[0], translation_matrices[0]])
    camera_pose_2 = np.hstack([rotation_matrices[1], translation_matrices[1]])
    camera_pose_3 = np.hstack([rotation_matrices[2], translation_matrices[2]])
    camera_pose_4 = np.hstack([rotation_matrices[3], translation_matrices[3]])
    
    # convert camera pose relative to base_pose
    camera_pose_1 = np.dot(camera_pose_1, base_pose)
    camera_pose_2 = np.dot(camera_pose_2, base_pose)
    camera_pose_3 = np.dot(camera_pose_3, base_pose)
    camera_pose_4 = np.dot(camera_pose_4, base_pose)
    
    # linear triangulation to find best pose
    best_count = 0
    best_pose = camera_pose_1
    for camera_pose in [camera_pose_1, camera_pose_2, camera_pose_3, camera_pose_4]:
        
        # loop through each point correspondence
        count = 0
        for index in range(0, len(ptsLeft)):
            pointLeft = ptsLeft[index]
            pointRight = ptsRight[index]
            
            # perform linear triangulation
            point = get_linear_triangulation(base_pose, camera_pose, pointLeft, pointRight, k_matrix)
            
            # check in front of the camera
            if(is_point_in_front(base_pose, point) and is_point_in_front(camera_pose, point)):
                count = count + 1
                
        # update best_pose found
        if(count > best_count):
            best_count = count
            best_pose = camera_pose
            
    # return best camera pose
    return best_pose
