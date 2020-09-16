# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 21:13:01 2020

@author: fubar
"""

import os, os.path
import numpy as np

import cv2



from fitEllipse import FitEllipse_LeastSquares, ConfidenceInFit, OverlayRANSACFit

from globalVariables import debug, deep_debug
from globalVariables import proc_dir, imgs_folder, samplename
from globalVariables import displayImages, saveImages
from globalVariables import nucleation_down, center_ROI, adaptive_ROI
from globalVariables import max_norm_err_sq

from processROI import processROI

from routinesOpenCV import processImage, removeSpeckles, cannyEdgesAndContoursOpenCV

from routinesMatplotLib import displayImage, displayNormError




def OverlayFitEllipse(img_edges, pnts, norm_err, inliers, best_ellipse):
    """Overlay the fit ellipse, inliers and outliers and return the image"""
    #create a color image
    img_color = cv2.merge((img_edges,img_edges,img_edges))
    if deep_debug:print(f"                img_color.shape: {img_color.shape}")
    OverlayRANSACFit(img_color, pnts, inliers, best_ellipse)
    if displayImages:
        displayNormError(img_color, norm_err)

    return img_color




def fitEllipse(img_edges,  ROI):
    """
        0. Filter the edges indices from the image
        1. Fit the edge to ellipse using mrgaze routine FitEllipse_LS
        2. Find the outliers, inliers and percent_inliers 
    """
    
    pnts = np.transpose(np.asarray(np.nonzero(np.transpose(img_edges))))
    
    ######################
    ## Least Squares Fit##
    ######################
    
    best_ellipse =  FitEllipse_LeastSquares(pnts)

    #find the center coordinates in old coordinate system
    
    abs_center = np.array(best_ellipse[0]) + np.array(center_ROI) - 0.5*np.array(ROI,dtype=np.float)
    #Find confidence of fit
    perc_inliers, inliers, norm_err = ConfidenceInFit(pnts, best_ellipse, max_norm_err_sq, debug)
    #parameters to return as an 6-channel array element
    pulse = [perc_inliers, abs_center[1], abs_center[0], best_ellipse[1][1], best_ellipse[1][0],best_ellipse[2]]
    img_color = OverlayFitEllipse(img_edges, pnts, norm_err, inliers, best_ellipse)
    
    return pulse, img_color

def findEdge(pulse_index, iter_index, exp_index, img, parent_dir_abs):
    """
    Take as input image array and return the params of
    ellipse fit and the image with fit ellipse overlayed onto
    the original image
    """
    

    #set default ROI corresponding to clipping the bottom scale information
    ROI = [1022, 1344]
    
    if adaptive_ROI:
        #find optimum ROI
        ROI = processROI(pulse_index, iter_index, exp_index, img, parent_dir_abs) 

    # process the image
    img_med = processImage(img, ROI)
    
    # Otsu's thresholding
    ret, img_otsu = cv2.threshold(img_med,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    img_speckless = removeSpeckles(img_otsu)  #remove speckles

    img_edges = cannyEdgesAndContoursOpenCV(img_speckless, nucleation_down) #canny edge detection using openCV
  
    pulse, img_color = fitEllipse(img_edges,  ROI)
    
    return pulse, img_color




def saveImage(pulse_index, iter_index, exp_index, img_color, parent_dir_abs):
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

    #save the image if saveImages flag is True
    if saveImages:
        
        if debug: print("            L3 saveImage() saving image")
        
        proc_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, proc_dir))
        if not os.path.isdir(proc_dir_abs): os.mkdir(proc_dir_abs)
        
        # image fits folder root
        img_dir_abs = os.path.abspath(os.path.join(proc_dir_abs, imgs_folder))
        img_root = os.path.abspath(os.path.join(img_dir_abs,samplename))
        
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
        
        
def processOpenCV(pulse_index, iter_index, exp_index, img_file, parent_dir_abs):
    """
        0. Read the image
        1. Fit the Ellipse
        2. Save the image
    """
    #Read the image file
    img = cv2.imread(img_file, 0)
    
    # Fit the ellise
    pulse, img_color = findEdge(pulse_index, iter_index, exp_index, img, parent_dir_abs)
    
    #save the image to file    
    saveImage(pulse_index, iter_index, exp_index, img_color, parent_dir_abs)
    
    return pulse