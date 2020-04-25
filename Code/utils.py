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
import glob
import cv2
import numpy as np


# get the grayscale image after converting it to lab colorspace
def get_grayscale_image(image):
    """
    Inputs:
    
    image: the input color image
    
    Outputs:
    
    gray: grayscale image
    """
    
    # convert to lab space, apply clahe to b channel and then get grayscale image
    #clahe = cv2.createCLAHE(clipLimit = 1.0, tileGridSize = (1, 1))
    #lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    #l, a, b = cv2.split(lab)
    #lab = cv2.merge((l, a, clahe.apply(b)))
    #lab = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 5)    

    # return grayscale image
    return gray

def crop_warped(image,rect):
    
    """
    Inputs:
    image: The warped image after performing an affine transformation
    rect: The upleft and the truth bounding box
    
    """
    #warped = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    warped = image[rect[1]:rect[3], rect[0]:rect[2]]
    return warped

# update the grayscale image, that is, normalize the pixel values with template image
def update_grayscale_image(template_image, image):
    """
    Inputs:
    
    template_image: the template grayscale image whose ROI is given to us
    image: the current grayscale frame 
    
    Outputs:
    
    image: the normalized current grayscale frame
    """
    
    # get the mean of the template image and current frame and normalize
    template_mean = np.mean(template_image)
    mean = np.mean(image)
    image = (image * (template_mean / mean)).astype(float)
    
    # return the normalized current grayscale frame
    return image


# lucas-kanade algorithm
def lucas_kanade_algo(template_frame, current_frame, x_range, y_range, p, thresh, delta_p_constant, flag, add_brightness_weight):
    
    # compute the roi of the template
    template_frame = template_frame[int(y_range[0]):int(y_range[1]), int(x_range[0]):int(x_range[1])]
    
    # compute derivatives around x and y directions for the current frame
    sobelx = cv2.Sobel(current_frame, cv2.CV_64F, 1, 0, ksize = 5)
    sobely = cv2.Sobel(current_frame, cv2.CV_64F, 0, 1, ksize = 5)  

    count = 0
    while(count <= 50):
        count = count + 1
        
        # affine matrix
        affine_mat = np.array([[1 + p[0][0], p[2][0], p[4][0]], [p[1][0], 1 + p[3][0], p[5][0]]], dtype = np.float32)
        
        # warp the image
        warped_current_frame = cv2.warpAffine(current_frame, affine_mat, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        warped_current_frame = warped_current_frame[int(y_range[0]):int(y_range[1]), int(x_range[0]):int(x_range[1])]
        
        # warp the sobelx and sobely
        warped_sobelx = cv2.warpAffine(sobelx, affine_mat, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        warped_sobely = cv2.warpAffine(sobely, affine_mat, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        warped_sobelx = warped_sobelx[int(y_range[0]):int(y_range[1]), int(x_range[0]):int(x_range[1])]
        warped_sobely = warped_sobely[int(y_range[0]):int(y_range[1]), int(x_range[0]):int(x_range[1])]
        warped_sobelx = warped_sobelx.reshape(-1, 1)
        warped_sobely = warped_sobely.reshape(-1, 1)
        
        # calculate error value
        error = (template_frame.astype(int) - warped_current_frame.astype(int)).reshape(-1, 1)
        
        # calculate steep descent value
        count = 0
        steep_descent = []
        for y in range(y_range[0], y_range[1]):
            for x in range(x_range[0], x_range[1]):
                jacobian = [x * warped_sobelx[count][0], x * warped_sobely[count][0], y * warped_sobelx[count][0], y * warped_sobely[count][0], warped_sobelx[count][0], warped_sobely[count][0]]
                steep_descent.append(jacobian)

                if(add_brightness_weight and (error[count][0] < -50 or error[count][0] > 50)):
                    error[count][0] = 0

                count = count + 1
        steep_descent = np.array(steep_descent)
        
        # calculate the hessian matrix and its inverse
        sd_param_matrix = np.dot(steep_descent.T, error)
        hessian_matrix = np.dot(steep_descent.T, steep_descent)
        hessian_matrix_inv = np.linalg.pinv(hessian_matrix)
        
        # calculate delta_p and update p matrix
        delta_p = np.dot(hessian_matrix_inv, sd_param_matrix)
        p_norm = np.linalg.norm(delta_p)
        p = np.reshape(p, (6, 1))
        delta_p = delta_p_constant * delta_p
        p = p + delta_p
        
        if(flag and p_norm < thresh):
            break
        
    # return the updated p and rectangle cooridnates
    affine_mat = np.array([[1 + p[0][0], p[2][0], p[4][0]], [p[1][0], 1 + p[3][0], p[5][0]]], dtype = np.float32)
    top_left = np.array([[x_range[0]], [y_range[0]], [1]])
    bottom_right = np.array([[x_range[1]], [y_range[1]], [1]])
    updated_top_left = np.dot(affine_mat, top_left)
    updated_bottom_right = np.dot(affine_mat, bottom_right)
    return (p, updated_top_left, updated_bottom_right)

# The inverse Compositional Lucas Kanade Algorithm
def LucasKanadeInverseCompositional(template, image, rect, affine_matrix, threshold, iterations):
    
    """
    Inputs:
    
    image: The input gray scale image
    template: The template image
    rect: The top left coordinates and the bottom right coordinates of the rectangle format--> [x1,y1,x2,y2]
    threshold: The threshold that has been set to terminate the iterations as soon as delta p value approaches the threshold value
    parameters: The initial parameters of the affine warp
    iterations: Number of times the algorithm needs to run
    
    Returns:
    
    p: Paramters of the affine warp matrix
    Upper left bounding bax coordinates
    Downright bounding box coordinates
    
    """
    # Threshold for convergence
    thresh = threshold
    
    # Crop the Region of interest in our template
    template = crop_warped(template, rect)
    
    # Evaluation of the template gradient 
    # Compute the image gradient in the x direction
    sobelx_template = cv2.Sobel(template, cv2.CV_64F, dx = 1, dy = 0, ksize = 5)
        
    # Compute the image gradient in the y direction
    sobely_template = cv2.Sobel(template, cv2.CV_64F, dx = 0, dy = 1, ksize = 5)
    
    # Horizontally stack the obtained Template gradients
    # Flatten out the warped gradients
    sobelx_template = sobelx_template.reshape(-1,1)
    sobely_template = sobely_template.reshape(-1,1)
    
    # Template Gradient
    template_gradient = np.hstack((sobelx_template,sobely_template))
    
    I = affine_matrix
    
    # Evaluate the Jacobian and the steepest descent 
    count = 0
    steepest_descent = []
    for y in range(rect[1],rect[3]):
        for x in range(rect[0],rect[2]):
                
            Jacobian = [x*sobelx_template[count][0], x*sobely_template[count][0], y*sobelx_template[count][0], y*sobely_template[count][0], sobelx_template[count][0], sobely_template[count][0]]
            steepest_descent.append(Jacobian)
            count = count + 1
    steepest_descent = np.array(steepest_descent)
    
    # Evaluate the Hessian Matrix
    hessian = np.matmul(steepest_descent.T,steepest_descent)
    
    for iterations in range(iterations):
        
        # Warp the Input Image
        warped_image = cv2.warpAffine(image, affine_matrix, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        
        # Compute the error image
        error_image = crop_warped(warped_image,rect) - template
        
        # Update the steepest descent parameters
        sd_params_update = np.matmul(steepest_descent.T, error_image.reshape(-1,1))
        
        #Compute delta p
        delta_p = np.matmul(np.linalg.pinv(hessian),sd_params_update)
        
        #Compute the updated Warp matrix
        affine_warp_delta_p = np.vstack([delta_p.reshape(2,3) + I,[0,0,1]])
        updated_warp_matrix = np.matmul(affine_matrix, np.linalg.inv(affine_warp_delta_p))
        
        # Convergence test
        if np.linalg.norm(delta_p) <= thresh:
            break
    top_left_coordinates = np.array([[rect[0]],[rect[1]],[1]])
    bottom_right_coordinates = np.array([[rect[2]], [rect[3]], [1]])
    updated_top_left_coordinates = np.matmul(updated_warp_matrix, top_left_coordinates)
    updated_bottom_right_coordinates = np.matmul(updated_warp_matrix, bottom_right_coordinates)
    
    #print(f'top_left: {updated_top_left_coordinates} \tbottom_right:{updated_bottom_right_coordinates}')
    return updated_warp_matrix, updated_top_left_coordinates, updated_bottom_right_coordinates
