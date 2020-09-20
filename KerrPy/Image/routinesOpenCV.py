# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:08:29 2020

@author: fubar
"""
import cv2
import numpy as np


from globalVariables import center_ROI, nucleation_down, speckless_second

from globalVariables import kernel_gaussian, kernel_median, kernel_speckless, kernel_speckless_second


def readImage(img_file):
    """
        Read the img_file
        with flag 0 (for grayscale)
        
        For more on imread modes/flags,
        Reference
        Imread modes
        https://docs.opencv.org/trunk/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
        
        Return a 2D array
        
    """
    #
    img = cv2.imread(img_file, 0)
    
    return img

def writeImage(img_file, img):
    """
        write img to img_file
    """
    cv2.imwrite(img_file, img)

def originROI(ROI):
    """
        Return the origin of ROI 
        i.e. top left corner
    """
    x_width, y_width    = ROI
    x_center, y_center  = center_ROI 
    
    x_origin   = int(x_center - x_width//2)
    y_origin   = int(y_center - y_width//2)
    
    origin = [x_origin, y_origin]
    
    return origin

def cropImageDetails(img):
    """
        Strip the image details at the bottom.
        The strip is width 44, 1344
    """
    return img[ 0:-44, :]


def cropImage(img, ROI):
    """
        0. Find the origin of the crop
        1. Crop the image
    """
    origin = originROI(ROI)
    
    img_crop = img[origin[0] : origin[0] + ROI[0], origin[1] : origin[1] + ROI[1]]
    
    return img_crop

def processImage(img, ROI):
    """"
        0. Crop the image
        1. Histogram equalize the image
        2. Gaussian blur the image
        3. Median filter the image
        
        Use the global training parameters
        kernel_Gaussian
        kernel_median
    """
    img_crop = cropImage(img, ROI)
    img_equ = cv2.equalizeHist(img_crop)
    img_blur = cv2.GaussianBlur(img_equ,(kernel_gaussian,kernel_gaussian),0)
    img_med= cv2.medianBlur(img_blur,kernel_median)
    
    return img_med

def removeSpeckles(img):
    """
        0. opening and
        1. closing the image
        to remove the speckles in background
        and holes in object
        
        If the nucleation is down, first close then open the image.
        else reverse the order of transformation.
        
        Reference: Morphological transformations with openCV
        https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
        
        # TODO check if the order of opening and closing affects
        removal of speckless differently for nucleation up and down images
         
    """

    img_speckless = np.zeros(img.shape)

    # first level of morph transformation
        
    kernel = np.ones((kernel_speckless,kernel_speckless),np.uint8)

    
    if nucleation_down:
                
        
        # close the image
        
        img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
        # open the image 
        
        img_closedthenopened = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel)
        
        img_speckless = img_closedthenopened
        
    else:
    
        # open the image
        
        img_opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
        # close the image 
        img_openedthenclosed = cv2.morphologyEx(img_opened, cv2.MORPH_CLOSE, kernel)
    
        img_speckless = img_openedthenclosed
        
    if speckless_second:
    
        # second level of morph transformation
        
        kernel_second = np.ones((kernel_speckless_second, kernel_speckless_second),np.uint8)
    
        if nucleation_down:
            
            # close the image
            
            img_closedthenopenedthenclosed = cv2.morphologyEx(img_closedthenopened, cv2.MORPH_CLOSE, kernel_second)
            
            # open the image
            
            img_closedthenopenedthenclosedthenopened = cv2.morphologyEx(img_closedthenopenedthenclosed, cv2.MORPH_OPEN, kernel_second)
            
            img_speckless = img_closedthenopenedthenclosedthenopened
            
        else:
            
            # open the image
            
            img_openthenclosedthenopened = cv2.morphologyEx(img_openedthenclosed, cv2.MORPH_OPEN, kernel_second)
            
            # close the image
            
            img_openthenclosedthenopenedthenclosed = cv2.morphologyEx(img_openthenclosedthenopened, cv2.MORPH_CLOSE, kernel_second)
            
            img_speckless = img_openthenclosedthenopenedthenclosed
        
    return img_speckless

def cannyEdgesAndContoursOpenCV(image, nucleation_down, lower_threshold=100, upper_threshold=200):
    """
        0. find Canny edges using CV
        as it has advantage of removing final speckles
        if still present after opening and closing
    """
    from globalVariables import nucleation_down, deep_debug
    
    edges = []
    contours = []
    if nucleation_down:
        edges = cv2.Canny(image, lower_threshold, upper_threshold)
    else:
        edges = cv2.Canny(np.invert(image), lower_threshold, upper_threshold)

    if edges.any():
        #find the contours in the edges to check the number of connected componenets
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
    #contours is a list of connected contours
    if contours:
        n_contours = len(contours)
    else : 
        n_contours = 0
    if deep_debug: print(f"                n_edges: {n_contours}")
    
    
    # array for number of points in each contour
    
    array_contour_length = np.zeros(n_contours)

    for i in np.arange(n_contours):
        array_contour_length[i] = len(contours[i])

    if deep_debug: print(f"lengths of contours are: {array_contour_length}")
    # sort the array length wise to extract the largest contour!

    # find the indices of the sorted array        

    sorted_indices = np.argsort(array_contour_length)
    
    if deep_debug: print(f"sorted_indices are: {sorted_indices}")

    # set the contour with largest length
    largest_contour = contours[sorted_indices[-1]]
    
    if deep_debug: print(f"shape of largest_contour: {largest_contour.shape}")

    # initialze image for single contour
    img_single_contour = np.zeros(edges.shape, dtype= np.uint8)
    
    # set the pixel intensity to HIGH for points of the contour
    for i in np.arange(largest_contour.shape[0]):
        row = largest_contour[i][0][0]
        col = largest_contour[i][0][1]
        
        #interchange rows and columns as openCV coordinates
        # transpose of 2D array rows and columns
        img_single_contour[col][row] = 255
        
    
    return img_single_contour
