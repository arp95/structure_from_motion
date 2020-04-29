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


# header files
from utils import *
import glob
import numpy as np
import cv2
from ReadCameraModel import *
from UndistortImage import *


# get files
files = glob.glob("/home/arpitdec5/Desktop/structure_from_motion/data/stereo/centre/*")
files = sorted(files)

# run the algo
count = 0
base_pose = np.matrix(np.hstack([np.eye(3), np.zeros((3, 1))]))
base_pose = np.vstack([base_pose, np.matrix([0, 0, 0, 1], dtype=np.float)])
x_points = []
z_points = []
original_x_points = []
original_z_points = []
for index in range(0, len(files) - 1):
    
    # get two images and resize them
    (image1, k_matrix) = get_image(files[index])
    (image2, k_matrix) = get_image(files[index+1])
    image1 = cv2.resize(image1, dsize = (0, 0), fx = 0.25, fy = 0.25)
    image2 = cv2.resize(image2, dsize = (0, 0), fx = 0.25, fy = 0.25)
    
    # convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # get keypoints
    (ptsLeft, ptsRight) = get_keypoints(gray1, gray2)
    
    # get fundamental matrix
    fundamental_matrix = get_fundamental_matrix_ransac(ptsLeft.copy(), ptsRight.copy())
    
    # get essential matrix (from opencv and without opencv)
    essential_matrix = get_essential_matrix(fundamental_matrix, k_matrix)
    original_essential_matrix, _ = cv2.findEssentialMat(np.array(ptsLeft), np.array(ptsRight), focal = k_matrix[0, 0], pp = (k_matrix[0, 2], k_matrix[1, 2]), method = cv2.RANSAC, prob = 0.999, threshold = 0.5)
    
    # get camera poses
    matrices = get_camera_poses(essential_matrix)
    translation_matrices = np.array(matrices[0])
    rotation_matrices = np.array(matrices[1])
    
    # get best pose (with opencv and without opencv)
    best_camera_pose = get_best_camera_pose(translation_matrices, rotation_matrices, base_pose, ptsLeft, ptsRight, k_matrix)
    best_camera_pose = np.vstack([best_camera_pose, np.matrix([0, 0, 0, 1], dtype=np.float)])
    _, original_rotation_matrix, original_translation_matrix, mask = cv2.recoverPose(original_essential_matrix, np.array(ptsLeft), np.array(ptsRight), focal = k_matrix[0, 0], pp = (k_matrix[0, 2], k_matrix[1, 2]))
    if(np.linalg.det(original_rotation_matrix) < 0):
        original_rotation_matrix = -original_rotation_matrix
        original_translation_matrix = -original_translation_matrix
    original_pose = np.hstack([original_rotation_matrix, original_translation_matrix])
    original_pose = np.vstack([original_pose, np.matrix([0, 0, 0, 1], dtype=np.float)])
    original_pose = np.dot(original_pose, base_pose)
    
    # update the base_pose for next frame
    x_points.append(best_camera_pose[0, 3])
    z_points.append(best_camera_pose[2, 3])
    original_x_points.append(original_pose[0, 3])
    original_z_points.append(original_pose[2, 3])
    base_pose = best_camera_pose
    
    count = count + 1
    if(count > 5):
        break
