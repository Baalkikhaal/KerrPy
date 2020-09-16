# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:08:29 2020

@author: fubar
"""
import cv2
import numpy as np


from globalVariables import center_ROI

from globalVariables import kernel_gaussian, kernel_median, kernel_speckless, kernel_speckless_second


def originROI(ROI):
    """
        Return the origin of ROI 
        i.e. top left corner
    """
    x_width, y_width    = ROI
    x_center, y_center  = center_ROI 
    
    x_origin   = int(x_center -x_width/2)
    y_origin   = int(y_center -y_width/2)
    
    origin = [x_origin, y_origin]
    
    return origin

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
        
        Reference: Morphological transformations with openCV
        https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
    """
    kernel = np.ones((kernel_speckless,kernel_speckless),np.uint8)

    # erode the image
    img_opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # dilate the image 
    img_openedthenclosed = cv2.morphologyEx(img_opened, cv2.MORPH_CLOSE, kernel)

    img_speckless = img_openedthenclosed

    kernel_second = np.ones((kernel_speckless_second, kernel_speckless_second),np.uint8)
    img_openthenclosedthenopened = cv2.morphologyEx(img_openedthenclosed, cv2.MORPH_OPEN, kernel_second)
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
    if nucleation_down == 1:
        edges = cv2.Canny(image, lower_threshold, upper_threshold)
    else:
        edges = cv2.Canny(np.invert(image), lower_threshold, upper_threshold)

    if edges.any():
        #find the contours in the edges to check the number of connected componenets
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
    #contours is a list of connected contours
    if contours:
        n_edges = len(contours)
    else : 
        n_edges = 0
    if deep_debug: print(f"                n_edges: {n_edges}")
    
    # largest_contour = contours[0]
    # pnts = np.asarray(largest_contour[:,0])

    # collect the contours which are greater than
    # minimum contour length
    
    return edges
