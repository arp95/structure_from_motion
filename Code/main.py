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
for index in range(0, len(files) - 1, 2):
    
    # get two images
    (image1, k_matrix) = get_image(files[index])
    (image2, k_matrix) = get_image(files[index+1])
    
    # get keypoints
    (ptsLeft, ptsRight) = get_keypoints(image1, image2)
    
    # get fundamental matrix
    fundamental_matrix = get_fundamental_matrix_ransac(ptsLeft.copy(), ptsRight.copy())
    
    # get essential matrix
    essential_matrix = get_essential_matrix(fundamental_matrix, k_matrix)
    
    count = count + 1
