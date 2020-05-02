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
from scipy.optimize import leastsq
from matplotlib import pyplot as plt


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
    fx, fy, cx, cy, camera_image, LUT = ReadCameraModel("data/model/")
    k_matrix = np.zeros((3, 3))
    k_matrix[0, 0] = fx
    k_matrix[1, 1] = fy
    k_matrix[2, 2] = 1
    k_matrix[0, 2] = cx
    k_matrix[1, 2] = cy
    image = UndistortImage(image, LUT)
    return (image, k_matrix)


#  get transformation matrices
def get_transformation_matrix(pl, pr):
    
    # mean for ptsLeft and ptsRight
    ptsLeft_mean_x = np.mean(pl[:, 0])
    ptsLeft_mean_y = np.mean(pl[:, 1])
    ptsRight_mean_x = np.mean(pr[:, 0])
    ptsRight_mean_y = np.mean(pr[:, 1])
    
    # scale factor for ptsLeft and ptsRight
    scale_ptsLeft = np.sqrt(2) / np.sum(((pl[:, 0] - ptsLeft_mean_x) ** 2 + (pl[:, 1] - ptsLeft_mean_y) ** 2) ** (1 / 2))
    scale_ptsRight = np.sqrt(2) / np.sum(((pr[:, 0] - ptsRight_mean_x) ** 2 + (pr[:, 1] - ptsRight_mean_y) ** 2) ** (1 / 2))
    
    # get transformation matrices
    ptsLeft_transformation_matrix = np.dot(np.array([[scale_ptsLeft, 0, 0], [0, scale_ptsLeft, 0], [0, 0, 1]]), np.array([[1, 0, -ptsLeft_mean_x], [0, 1, -ptsLeft_mean_y], [0, 0, 1]]))
    ptsRight_transformation_matrix = np.dot(np.array([[scale_ptsRight, 0, 0], [0, scale_ptsRight, 0], [0, 0, 1]]), np.array([[1, 0, -ptsRight_mean_x], [0, 1, -ptsRight_mean_y], [0, 0, 1]]))
    
    # get normalized points
    for index in range(0, len(pl)):
        pl[index][0] = (pl[index][0] - ptsLeft_mean_x) * scale_ptsLeft
        pl[index][1] = (pl[index][1] - ptsLeft_mean_y) * scale_ptsLeft
        
    for index in range(0, len(pr)):
        pr[index][0] = (pr[index][0] - ptsRight_mean_x) * scale_ptsRight
        pr[index][1] = (pr[index][1] - ptsRight_mean_y) * scale_ptsRight
    
    # return matrices
    return (pl, pr, ptsLeft_transformation_matrix, ptsRight_transformation_matrix)


# get keypoints between frame 1 and frame 2
def get_keypoints(image1, image2):
    """
    Inputs:
    
    image1: left image
    image2: right image
    
    Outputs:
    
    pl: point correspondences for left image
    pr: point correspondences for right image
    """
    
    # use sift keypoint to get the points
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1,None)
    kp2, des2 = sift.detectAndCompute(image2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pl = []
    pr = []

    for i,(m,n) in enumerate(matches):
        if m.distance < (0.5*n.distance):
            pl.append(kp1[m.queryIdx].pt)
            pr.append(kp2[m.trainIdx].pt)
    return (np.array(pl), np.array(pr))


# get fundamental matrix with ransac
def get_fundamental_matrix_ransac(pl, pr):
    """
    Inputs:
    
    pl: array of 8 points for left image
    pr: array of 8 points for right image
    
    Outputs:
    
    fundamental_mat: fundamental matrix of size (3 x 3)
    """
    
    # normalise points
    (pl, pr, ptsLeft_transformation_matrix, ptsRight_transformation_matrix) = get_transformation_matrix(pl, pr)

    # ransac for better matrix estimation
    iterations = 2000
    threshold = 0.06
    count = 0
    best_ptsLeft = []
    best_ptsRight = []
    best_fundamental_matrix = get_fundamental_matrix(pl[:8], pr[:8], ptsLeft_transformation_matrix, ptsRight_transformation_matrix)
    for iteration in range(0, iterations):
        
        indexes = np.random.randint(len(pl), size = 8)
        random_ptsLeft = np.array([pl[indexes[0]], pl[indexes[1]], pl[indexes[2]], pl[indexes[3]], pl[indexes[4]], pl[indexes[5]], pl[indexes[6]], pl[indexes[7]]])
        random_ptsRight = np.array([pr[indexes[0]], pr[indexes[1]], pr[indexes[2]], pr[indexes[3]], pr[indexes[4]], pr[indexes[5]], pr[indexes[6]], pr[indexes[7]]])
    
        estimated_fundamental_mat = get_fundamental_matrix(random_ptsLeft, random_ptsRight, ptsLeft_transformation_matrix, ptsRight_transformation_matrix)
        estimated_count = 0
        sample_ptsLeft = []
        sample_ptsRight = []
        for index in range(0, len(pl)):
            x_right = np.array([pr[index][0], pr[index][1], 1])
            x_left = np.array([pl[index][0], pl[index][1], 1]).T
            
            if(abs(np.squeeze(np.matmul((np.matmul(x_right, estimated_fundamental_mat)), x_left))) < threshold):
                estimated_count = estimated_count + 1
                sample_ptsLeft.append(pl[index])
                sample_ptsRight.append(pr[index])
                
        if(estimated_count > count):
            count = estimated_count
            best_fundamental_matrix = estimated_fundamental_mat
            best_ptsLeft = sample_ptsLeft
            best_ptsRight = sample_ptsRight
    
    # return fundamental matrix
    return (best_fundamental_matrix, np.array(best_ptsLeft), np.array(best_ptsRight))
    
    
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
    #fundamental_mat = fundamental_mat / np.linalg.norm(fundamental_mat)
    
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
    #essential_matrix = essential_matrix / np.linalg.norm(essential_matrix)
    
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
    u, d, v = np.linalg.svd(essential_matrix, full_matrices=True)

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
    if((np.dot(r[2, :], (point - t))) > 0):
        return True
    return False 


# performs linear triangulation
def get_linear_triangulation(camera_pose_1, camera_pose_2, pointLeft, pointRight):
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
    m_matrix = np.vstack([np.dot(pointLeft_cross_product, camera_pose_1[:3, :]), np.dot(pointRight_cross_product, camera_pose_2)])
    
    # get the 3D point
    u, s, vh = np.linalg.svd(m_matrix)
    point = vh[-1]
    point = (point / point[3]).reshape((4, 1))
    point = point[:3].reshape((3, 1))
    
    # return point
    return point
    
    
# performs non-linear triangulation
def get_non_linear_triangulation(camera_pose_1, camera_pose_2, pointLeft, pointRight):
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
    
    # perform linear triangulation and get linear estimate
    estimated_point = get_linear_triangulation(camera_pose_1, camera_pose_2, pointLeft, pointRight)
    
    # run Levenberg-Marquardt algorithm
    args = (camera_pose_1, camera_pose_2, pointLeft, pointRight)
    point, success = leastsq(get_triangulation_error, estimated_point, args = args, maxfev = 10000)
    point = np.matrix(point).T
    
    # return point
    return point


# the triangulation error function for non-linear triangulation
def get_triangulation_error(estimated_point, camera_pose_1, camera_pose_2, pointLeft, pointRight):

    # project into each frame
    estimated_point = np.array([estimated_point[0, 0], estimated_point[1, 0], estimated_point[2, 0], [1]])
    estimated_ptLeft = fromHomogenous(np.dot(camera_pose_1, estimated_point))
    estimated_ptRight = fromHomogenous(np.dot(camera_pose_2, estimated_point))
    estimated_ptLeft = np.array([estimated_ptLeft[0, 0] / estimated_ptLeft[2, 0], estimated_ptLeft[0, 0] / estimated_ptLeft[1, 0]])
    estimated_ptRight = np.array([estimated_ptRight[0, 0] / estimated_ptRight[2, 0], estimated_ptRight[0, 0] / estimated_ptRight[1, 0]])
    
    # compute the diffs
    diff1 = estimated_ptLeft - pointLeft
    diff2 = estimated_ptRight - pointRight

    # return error
    return np.asarray(np.vstack([diff1, diff2]).T)[0, :]
    
    
# estimate the best camera pose
def get_best_camera_pose(translation_matrices, rotation_matrices, base_pose, ptsLeft, ptsRight):
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
            point = get_linear_triangulation(base_pose, camera_pose, pointLeft, pointRight)
            
            # check in front of the camera
            if(is_point_in_front(camera_pose, point)):
                count = count + 1
                
        # update best_pose found
        if(count > best_count):
            best_count = count
            best_pose = camera_pose
            
    # assuming motion is forward
    #if(best_pose[2, 3] > 0):
    #    best_pose[2, 3] = -best_pose[2, 3]
            
    # return best camera pose
    return best_pose
