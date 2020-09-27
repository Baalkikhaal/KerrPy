# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 21:13:01 2020

@author: fubar
"""

import os, os.path
import numpy as np

import cv2


from globalVariables import debug, deep_debug

from globalVariables import proc_dir, temp_dir, imgs_folder, restored_folder, samplename

from globalVariables import displayImages, saveImages, saveRestoredImages
from globalVariables import nucleation_down
from globalVariables import center_ROI, adaptive_ROI, aspr_ROI, adaptive_ROI_seq, custom_ROI
from globalVariables import max_norm_err_sq


from KerrPy.Image.processROI import processROI, restoreColorROI, restoreColorCustomROI

from KerrPy.Image.routinesOpenCV import processImage, removeSpeckles, cannyEdgesAndContoursOpenCV

from KerrPy.Image.routinesOpenCV import processImageCustomROI

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




def fitEllipse(img_edges,  ROI):
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
    perc_inliers, inliers, norm_err = ConfidenceInFit(pnts, best_ellipse, max_norm_err_sq, debug)
    
    int_perc_inliers = np.int(perc_inliers)
    
    # save parameters in absolute coordinate system
    rel_ellipse_center = np.array(best_ellipse[0], dtype = np.int)
    rel_ellipse_major, rel_ellipse_minor = np.array(best_ellipse[1], dtype = np.int)
    rel_ellipse_orientation = np.array(best_ellipse[2], dtype = np.int)
    
    #find the center coordinates in absolute coordinate system
    
    # origin of ROI
    origin_ROI = center_ROI - 0.5*np.array(ROI)
    int_origin_ROI = origin_ROI.astype(int)
    
    # since we transposed the image, the coordinates of ellipse also need to be transposed
    rel_ellipse_center_transpose = np.array([rel_ellipse_center[1], rel_ellipse_center[0]])
    
    abs_ellipse_center = int_origin_ROI + rel_ellipse_center_transpose
    
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

def fitEllipseCustomROI(img_edges,  customROI):
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
    perc_inliers, inliers, norm_err = ConfidenceInFit(pnts, best_ellipse, max_norm_err_sq, debug)
    
    int_perc_inliers = np.int(perc_inliers)
    
    # save parameters in absolute coordinate system
    rel_ellipse_center = np.array(best_ellipse[0], dtype = np.int)
    rel_ellipse_major, rel_ellipse_minor = np.array(best_ellipse[1], dtype = np.int)
    rel_ellipse_orientation = np.array(best_ellipse[2], dtype = np.int)
    
    #find the center coordinates in absolute coordinate system
    
    # origin of customROI
    origin_customROI = np.array([customROI[0], customROI[1]])
    int_origin_customROI = origin_customROI.astype(int)
    
    # since we transposed the image, the coordinates of ellipse also need to be transposed
    rel_ellipse_center_transpose = np.array([rel_ellipse_center[1], rel_ellipse_center[0]])
    
    abs_ellipse_center = int_origin_customROI + rel_ellipse_center_transpose
    
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


def findEdge(pulse_index, iter_index, exp_index, img, parent_dir_abs):
    """
    Take as input image array and return the params of
    ellipse fit and the image with fit ellipse overlayed onto
    the original image
    """
    

    #set default ROI corresponding to clipping the bottom scale information
    ROI = np.array([adaptive_ROI_seq[pulse_index], aspr_ROI * adaptive_ROI_seq[pulse_index]], dtype = np.int)
    
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

def findEdgeCustomROI(coords, pulse_index, iter_index, exp_index, img, parent_dir_abs):

    # interchange rows and columns to reflect coordinate transpose of openCV and numpy
    customROI= np.array([coords[1][0], coords[0][0], coords[1][2], coords[0][2]])

    customROI = customROI.astype(int)
    
    print(f"customROI: {customROI}")

    # process the image with custom ROI
    img_med = processImageCustomROI(img, customROI)

    # Otsu's thresholding
    ret, img_otsu = cv2.threshold(img_med,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    img_speckless = removeSpeckles(img_otsu)  #remove speckles

    img_edges = cannyEdgesAndContoursOpenCV(img_speckless, nucleation_down) #canny edge detection using openCV
  
    pulse, img_color = fitEllipseCustomROI(img_edges,  customROI)

        
    return pulse, img_color, customROI


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

    
    if debug: print("            L3 saveImage() saving image")

    proc_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, proc_dir))
    if not os.path.isdir(proc_dir_abs): os.mkdir(proc_dir_abs)


    # image fits folder root
    img_dir_abs = os.path.abspath(os.path.join(proc_dir_abs, imgs_folder))
    if not os.path.isdir(img_dir_abs): os.mkdir(img_dir_abs)

 

    img_root = os.path.abspath(os.path.join(img_dir_abs,samplename))
    if not os.path.isdir(img_root): os.mkdir(img_root)

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
 
def saveRestoredImage(pulse_index, iter_index, exp_index, img_restored, parent_dir_abs):
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
    
    proc_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, proc_dir))
    if not os.path.isdir(proc_dir_abs): os.mkdir(proc_dir_abs)
    
    # image fits folder root
    restored_dir_abs = os.path.abspath(os.path.join(proc_dir_abs, restored_folder))
    if not os.path.isdir(restored_dir_abs): os.mkdir(restored_dir_abs)

    restored_root = os.path.abspath(os.path.join(restored_dir_abs,samplename))
    if not os.path.isdir(restored_root): os.mkdir(restored_root)

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


        
def processOpenCV(pulse_index, iter_index, exp_index, img_file, parent_dir_abs):
    """
        0. Read the image
        1. Fit the Ellipse
        2. Save the image
    """
    #Read the image file
    img = cv2.imread(img_file, 0)
    
    
    if adaptive_ROI:    
        # Fit the ellise
        pulse, img_color = findEdge(pulse_index, iter_index, exp_index, img, parent_dir_abs)

    elif custom_ROI:
        # read the coordinates of CustomROI from "coordinates.npy" at parent dir
        
        coords_file = os.path.abspath(os.path.join(parent_dir_abs, temp_dir, 'coordinates.npy'))
    
        coords  = np.load(coords_file)
        # Fit the ellise with rectROI
        pulse, img_color, customROI = findEdgeCustomROI(coords, pulse_index, iter_index, exp_index, img, parent_dir_abs)
        
    # initialize restored image
    img_restored = np.zeros(img.shape)
    
    if adaptive_ROI:
        # restored the colored fit onto the original image
        img_restored = restoreColorROI(img_color, img)
    
    elif custom_ROI:
        # restored the colored fit onto the original image with rectangle ROI
        img_restored = restoreColorCustomROI(img_color, img, customROI)        
    
    if saveImages:
        #save the image to file    
        saveImage(pulse_index, iter_index, exp_index, img_color, parent_dir_abs)

    # save the restored image to file
    
    if saveRestoredImages:
        
        saveRestoredImage(pulse_index, iter_index, exp_index, img_restored, parent_dir_abs)
    
    return pulse
