# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 21:13:01 2020

@author: fubar
"""

import os, os.path
import numpy as np

import cv2


from globalVariables import debug, deep_debug

from globalVariables import displayImages, saveImages, saveRestoredImages

from KerrPy.File.loadFilePaths import img_root, restored_root

from KerrPy.Image.processROI import processROI, restoreColorROI

from KerrPy.Image.routinesOpenCV import processImage, removeSpeckles, cannyEdgesAndContoursOpenCV


from KerrPy.Figure.routinesMatplotLib import displayImage, displayNormError

from KerrPy.Image.fitEllipse import FitEllipse_LeastSquares, ConfidenceInFit, OverlayRANSACFit





def OverlayFitEllipse(img_edges, pnts, norm_err, inliers, best_ellipse):
    """Overlay the fit ellipse, inliers and outliers and return the image"""
    #create a color image
    img_color = cv2.merge((img_edges,img_edges,img_edges))
    if deep_debug:print(f"                img_color.shape: {img_color.shape}")
    OverlayRANSACFit(img_color, pnts, inliers, best_ellipse)
    if displayImages:
        displayNormError(img_color, norm_err)

    return img_color


def fitEllipse(img_edges, windowROI):
    """
        0. Filter the edges indices from the image
        1. Fit the edge to ellipse using mrgaze routine FitEllipse_LS
        2. Find the outliers, inliers and percent_inliers 
    """
    # transpose the image to extract the points as coordinate system
    # for openCV is transpose of numpy array coordinate system
    # i.e. the point coordinates are transpose of array coordinates
    
    pnts = np.transpose(np.asarray(np.nonzero(np.transpose(img_edges))))
    
    ######################
    ## Least Squares Fit##
    ######################
    
    best_ellipse =  FitEllipse_LeastSquares(pnts)
    
    #Find confidence of fit
    perc_inliers, inliers, norm_err = ConfidenceInFit(pnts, best_ellipse)
    
    int_perc_inliers = np.int(perc_inliers)
    
    # save parameters in absolute coordinate system
    rel_ellipse_center = np.array(best_ellipse[0], dtype = np.int)
    rel_ellipse_major, rel_ellipse_minor = np.array(best_ellipse[1], dtype = np.int)
    rel_ellipse_orientation = np.array(best_ellipse[2], dtype = np.int)
    
    #find the center coordinates in absolute coordinate system
    
    # origin of windowROI
    origin_windowROI = np.array([windowROI[0], windowROI[1]])
    int_origin_windowROI = origin_windowROI.astype(int)
    
    # since we transposed the image, the coordinates of ellipse also need to be transposed
    rel_ellipse_center_transpose = np.array([rel_ellipse_center[1], rel_ellipse_center[0]])
    
    abs_ellipse_center = int_origin_windowROI + rel_ellipse_center_transpose
    
    abs_x_center = abs_ellipse_center[0]
    abs_y_center = abs_ellipse_center[1]
    
    # TODO also we need to flip the major and minor ??!! check for rotated ellipse
    
    abs_ellipse_major = rel_ellipse_minor
    abs_ellipse_minor = rel_ellipse_major
    
    # also we need to rotate the orientation by 180 modulo 360
    
    abs_ellipse_orientation = (rel_ellipse_orientation + 180 )%180
    
    
    #parameters to return as an 6-channel array element
    pulse = [int_perc_inliers, abs_x_center, abs_y_center, abs_ellipse_major, abs_ellipse_minor, abs_ellipse_orientation]
    array_pulse = np.array(pulse, dtype=np.float)
    
    # overlay the fit ellipse onto the image
    img_color = OverlayFitEllipse(img_edges, pnts, norm_err, inliers, best_ellipse)
    
    return array_pulse, img_color

def findEdge(pulse_index, iter_index, exp_index, img, **kwargs):
    """
    Take as input image array and return the params of
    ellipse fit and the image with fit ellipse overlayed onto
    the original image
    
    # TODO sub function ROI conditionals to processROI
    """
    
    # process ROI
    windowROI = processROI(pulse_index, iter_index, exp_index, img, **kwargs)
    
    # process the image
    img_med = processImage(img, windowROI)
    
    # Otsu's thresholding
    ret, img_otsu = cv2.threshold(img_med,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    img_speckless = removeSpeckles(img_otsu)  #remove speckles

    img_edges = cannyEdgesAndContoursOpenCV(img_speckless) #canny edge detection using openCV
  
    pulse, img_color = fitEllipse(img_edges, windowROI)

        
    return pulse, img_color, windowROI


def saveImage(pulse_index, iter_index, exp_index, img_color):
    """
        If global flag saveImages is True, then save the img_color
        at corresponding heirarchial location in Images roots directory
        
        Also if global flag displayImages is True, then show the images
        NOTE that turning this on may eat up the memory for displaying
        multiple images. So use only for debugging purposes!!!

    """

    if debug: print(f"            L3 saveImage() started at E:{exp_index} I:{iter_index} P:{pulse_index}")

    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)

    #display the image if displayImags flag is True
    if displayImages:
        title = "Ellipse Fit with Inliers and Outliers"
        displayImage(img_color, title) 

    
    if debug: print("            L3 saveImage() saving image")

    # change to Images Fits root folder (LEVEL 0)
    os.chdir(img_root)
    
    # image folder for current experiment; create it if not yet
    img_exp_folder = f"Experiment_{exp_index}"
    if not os.path.isdir(img_exp_folder): os.mkdir(img_exp_folder)
    
    #change to images experiment folder (LEVEL 1)
    os.chdir(img_exp_folder)

    # image folder for current iteration; create it if not yet
    img_iter_folder = f"Experiment_{exp_index}_Iteration_{iter_index}"
    if not os.path.isdir(img_iter_folder): os.mkdir(img_iter_folder)

    #change to images iteration folder (LEVEL 2)
    os.chdir(img_iter_folder)

    # image folder for current pulse; create it if not yet
    img_pulse_folder = f"Experiment_{exp_index}_Iteration_{iter_index}_Pulse_{pulse_index}"
    if not os.path.isdir(img_pulse_folder): os.mkdir(img_pulse_folder)
    
    #change to Fits pulse folder (LEVEL 3)
    os.chdir(img_pulse_folder)
    
    #save the image here
    img_file = f"{img_pulse_folder}.png"
    cv2.imwrite(img_file,img_color)

        
    #Restore the path to the image path
    os.chdir(cur_path)
 
def saveRestoredImage(pulse_index, iter_index, exp_index, img_restored):
    """
        If global flag restoreImages is True, then save the img_color
        at corresponding heirarchial location in Restored roots directory
        
        Also if global flag displayImages is True, then show the images
        NOTE that turning this on may eat up the memory for displaying
        multiple images. So use only for debugging purposes!!!

    """

    if debug: print(f"            L3 saveRestoredImage() started at E:{exp_index} I:{iter_index} P:{pulse_index}")

    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)

    #display the image if displayImags flag is True
    if displayImages:
        title = "Ellipse Fit with Inliers and Outliers"
        displayImage(img_restored, title) 

    
    if debug: print("            L3 saveRestoredImage() saving image")
    
    # change to Images Fits root folder (LEVEL 0)
    os.chdir(restored_root)
    
    # image folder for current experiment; create it if not yet
    restored_exp_folder = f"Experiment_{exp_index}"
    if not os.path.isdir(restored_exp_folder): os.mkdir(restored_exp_folder)
    
    #change to images experiment folder (LEVEL 1)
    os.chdir(restored_exp_folder)

    # image folder for current iteration; create it if not yet
    restored_iter_folder = f"Experiment_{exp_index}_Iteration_{iter_index}"
    if not os.path.isdir(restored_iter_folder): os.mkdir(restored_iter_folder)

    #change to images iteration folder (LEVEL 2)
    os.chdir(restored_iter_folder)

    # image folder for current pulse; create it if not yet
    restored_pulse_folder = f"Experiment_{exp_index}_Iteration_{iter_index}_Pulse_{pulse_index}"
    if not os.path.isdir(restored_pulse_folder): os.mkdir(restored_pulse_folder)
    
    #change to Fits pulse folder (LEVEL 3)
    os.chdir(restored_pulse_folder)
    
    #save the image here
    restored_file = f"{restored_pulse_folder}.png"
    cv2.imwrite(restored_file,img_restored)

        
    #Restore the path to the image path
    os.chdir(cur_path)


        
def processOpenCV(pulse_index, iter_index, exp_index, img_file):
    """
        0. Read the image
        1. Fit the Ellipse
        2. Save the image
    """
    #Read the image file
    img = cv2.imread(img_file, 0)
    
    # Fit the ellise
    pulse, img_color, windowROI = findEdge(pulse_index, iter_index, exp_index, img)

    # restored the colored fit onto the original image
    img_restored = restoreColorROI(img_color, img, windowROI)
    
    if saveImages:
        #save the image to file    
        saveImage(pulse_index, iter_index, exp_index, img_color)


    if saveRestoredImages:
        # save the restored image to file
        saveRestoredImage(pulse_index, iter_index, exp_index, img_restored)
    
    return pulse
