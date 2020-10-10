# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:02:03 2020

@author: fubar
"""

import os, os.path
import numpy as np

from globalVariables import debug, deep_debug
from globalVariables import dict_ROI
from globalVariables import displayImages, saveImages

from KerrPy.File.loadFilePaths import img_root

from KerrPy.Image.routinesOpenCV import processImage

from KerrPy.Statistics.routinesSciPy import modalAnalysis

from KerrPy.Figure.routinesMatplotLib import displayImage, plotROIanalysis


def saveROIAnalysis(pulse_index, iter_index, exp_index, plot_ROI_analysis_file, fig_ROI_analysis):
    """
        Take pulse_index, iter_index, exp_index, fig_mod_analysis as inputs
        and save as plots of ROI analysis in ROI_analysis folder
    """
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    if debug: print(f"                L4 saveROIAnalysis() started at E:{exp_index} I:{iter_index} P:{pulse_index}")

    #change to images root folder (LEVEL 0)
    os.chdir(img_root)

    # folder for current experiment; create it if not yet
    img_exp_folder = f"Experiment_{exp_index}"
    if not os.path.isdir(img_exp_folder): os.mkdir(img_exp_folder)
    
    #change to images experiment folder (LEVEL 1)
    os.chdir(img_exp_folder)

    # folder for current iteration; create it if not yet
    img_iter_folder = f"Experiment_{exp_index}_Iteration_{iter_index}"
    if not os.path.isdir(img_iter_folder): os.mkdir(img_iter_folder)

    #change to images iteration folder (LEVEL 2)
    os.chdir(img_iter_folder)

    # folder for current pulse; create it if not yet
    img_pulse_folder = f"Experiment_{exp_index}_Iteration_{iter_index}_Pulse_{pulse_index}"
    if not os.path.isdir(img_pulse_folder): os.mkdir(img_pulse_folder)
    
    #change to images pulse folder (LEVEL 3)
    os.chdir(img_pulse_folder)
    
    roi_analysis_folder = 'ROI_analysis'
    if not os.path.isdir(roi_analysis_folder): os.mkdir(roi_analysis_folder)
    
    #change to modal analysis folder
    os.chdir(roi_analysis_folder)

    #save the figure
    
    plot_ROI_analysis_file_png = f"{plot_ROI_analysis_file}.png"
    plot_ROI_analysis_file_pdf = f"{plot_ROI_analysis_file}.pdf"
    
    fig_ROI_analysis.savefig(plot_ROI_analysis_file_png)
    fig_ROI_analysis.savefig(plot_ROI_analysis_file_pdf)

    #Restore the path to the image path
    os.chdir(cur_path)
    
    
    
       

def saveModalAnalysis(pulse_index, iter_index, exp_index, plot_modal_analysis_file, fig_mod_analysis):
    """
        Take pulse_index, iter_index, exp_index, fig_mod_analysis as inputs
        and save as numpy array file at level 3 of fits in modal analysis folder
    """
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    if debug: print(f"                L4 saveModalAnalysis() started at E:{exp_index} I:{iter_index} P:{pulse_index}")

    #change to images root folder (LEVEL 0)
    os.chdir(img_root)

    # folder for current experiment; create it if not yet
    img_exp_folder = f"Experiment_{exp_index}"
    if not os.path.isdir(img_exp_folder): os.mkdir(img_exp_folder)
    
    #change to images experiment folder (LEVEL 1)
    os.chdir(img_exp_folder)

    # folder for current iteration; create it if not yet
    img_iter_folder = f"Experiment_{exp_index}_Iteration_{iter_index}"
    if not os.path.isdir(img_iter_folder): os.mkdir(img_iter_folder)

    #change to images iteration folder (LEVEL 2)
    os.chdir(img_iter_folder)

    # folder for current pulse; create it if not yet
    img_pulse_folder = f"Experiment_{exp_index}_Iteration_{iter_index}_Pulse_{pulse_index}"
    if not os.path.isdir(img_pulse_folder): os.mkdir(img_pulse_folder)
    
    #change to images pulse folder (LEVEL 3)
    os.chdir(img_pulse_folder)
    
    modal_analysis_folder = 'modal_analysis'
    if not os.path.isdir(modal_analysis_folder): os.mkdir(modal_analysis_folder)
    
    #change to modal analysis folder
    os.chdir(modal_analysis_folder)

    #save the figure
    plot_modal_analysis_file_png = f"{plot_modal_analysis_file}.png"
    plot_modal_analysis_file_pdf = f"{plot_modal_analysis_file}.pdf"
    
    fig_mod_analysis.savefig(plot_modal_analysis_file_png)
    fig_mod_analysis.savefig(plot_modal_analysis_file_pdf)

    #Restore the path to the image path
    os.chdir(cur_path)
    

def originROI(ROI):
    """
        Return the origin of ROI 
        i.e. top left corner
    """
    center_ROI = dict_ROI['center']
    x_width, y_width    = ROI
    x_center, y_center  = center_ROI 
    
    x_origin   = int(x_center - x_width//2)
    y_origin   = int(y_center - y_width//2)
    
    origin = [x_origin, y_origin]
    
    return origin

    
def getWindowROI(ROI):
    """
        Get the tuple of (x0, y0, x1, y1) of ROI
    """
    
    x_width = ROI[0]
    y_width = ROI[1]
    origin = originROI(ROI)
  
    x0 = origin[0]
    y0 = origin[1]
    x1 = x0 + x_width
    y1 = y0 + y_width
    
    return x0, y0, x1, y1
    
def findModes(pulse_index, iter_index, exp_index, img):
    """
        Zoom out of the ROI sequence and perform modal analysis
        on each ROI
    """
    
    # get the global dictionary keys
    aspr_ROI = dict_ROI['aspr']
    adaptive_ROI_seq = dict_ROI['seq']
    
    array_x_ROI     =   np.array(adaptive_ROI_seq)
    array_y_ROI     =   (array_x_ROI*aspr_ROI).astype(int)
    n           =   array_x_ROI.size

    # initialize the optimum ROI
    optimum_x_ROI   =  0
    optimum_y_ROI   =   0
    
    #set the array for relative strengths and maxima positions for the unimodal or bimodal distributions.
    array_rel_strength  =   np.zeros(n)
    array_maximum       =   np.zeros((n,2))  
    
    for i in np.arange(n):
        
        # adaptive ROI        
        ROI = [array_x_ROI[i], array_y_ROI[i]]
        
        # get the window of ROI
        windowROI = getWindowROI(ROI)
        
        # process the image
        img_med = processImage(img, windowROI)

        img_background = np.zeros([1022,1344], dtype = np.uint8)
        
        img_restored = restoreROI(img_med, img_background, windowROI)
        
        maximum,rel_strength, fig_mod_analysis    =   modalAnalysis(img_med, img_restored, ROI)    #strength is zero if distribution is unimodal and close to zero if the foreground is very small compared to background or vice versa
        
        if displayImages:
            if saveImages:
                plot_modal_analysis_file = f"ModalAnalysis_xROI = {ROI[0]} yROI = {ROI[1]}"
                saveModalAnalysis(pulse_index, iter_index, exp_index, plot_modal_analysis_file, fig_mod_analysis)
        array_rel_strength[i]   =   rel_strength   
        array_maximum[i]        =   maximum
    
    if displayImages:
        label = "xROI limited"
        fig_ROI_analysis = plotROIanalysis(array_y_ROI, array_rel_strength, label)
        
        if saveImages:
            plot_ROI_analysis_file = "ROIAnalysis_xROIlimited"
            saveROIAnalysis(pulse_index, iter_index, exp_index, plot_ROI_analysis_file, fig_ROI_analysis)
            
    # if all are unimodal distributions, then there either is no object
    # to be found or object is beyond the ROI.
    # This means that we need to check for bigger ROIs
    # with progressive increase in y axis width
    max_rel_strength = np.max(array_rel_strength)
    
    if debug: print('{}{:.2f}'.format('                max_rel_strength: ', max_rel_strength))
    
    #find the optimum ROI from maximum of the relative strength vs ROI variation
    optimum_x_ROI = array_x_ROI[array_rel_strength.argsort()[-1]]
    optimum_y_ROI = array_y_ROI[array_rel_strength.argsort()[-1]]
    optimum_ROI = (optimum_x_ROI,optimum_y_ROI)
    
    if debug: print(f"                Opt ROI: {optimum_ROI}")
            
    return optimum_ROI
            
    

def findAdaptiveROI(pulse_index, iter_index, exp_index, img):
    """
        Using Kernel Density Estimate analysis,
        relative amount of foreground and background regions
        within a given ROI can be estimated.

        If the roi is varied across the viewing area,
        an optimum relative strength can be found
        when the foreground and background area are nearly equal.
        This sets the value of adaptive threshold.
        
        Since we know certain properties about our object of interest
        like the approximate center of object
        and the aspect ratio of object area, the ROI can be set accordingly
    """

    # get the global dictionary keys
    isAdaptive = dict_ROI['isAdaptive']
    aspr_ROI = dict_ROI['aspr']
    adaptive_ROI_seq = dict_ROI['seq']
    
    
    if isAdaptive:
        
        optimum_ROI = findModes(pulse_index, iter_index, exp_index, img)
        #optimum windowROI is
        optimum_windowROI = getWindowROI(optimum_ROI)
    
    else:
        
        #set default ROI corresponding to maximum ROI of the given seq
        optimum_ROI = np.array([adaptive_ROI_seq[-1], aspr_ROI * adaptive_ROI_seq[-1]], dtype = np.int)
        optimum_windowROI = getWindowROI(optimum_ROI)
        
    
    return optimum_windowROI


def restoreROI(cropimg, img_background, windowROI):
    ''' Restore the cropped image onto the 1022 x 1344 canvas'''
    
    # origin of the ROI in absolute coordinates
    x0 = windowROI[0]
    y0 = windowROI[1]
    
    # end of the ROI in absolute coordinates
    x1 = windowROI[2]
    y1 = windowROI[3]
    
    img_restored = img_background
    
    #restore the cropped image
    img_restored[x0:x1, y0:y1] = cropimg
    
    if displayImages:
        title = 'Restored image'
        displayImage(img_restored, title)
    return img_restored

def restoreColorROI(img_color, img_background, windowROI):
    """
        Restore the colored crop image with ellipse fit
        
        onto the background
    """
    # origin of the ROI in absolute coordinates
    x0 = windowROI[0]
    y0 = windowROI[1]
    
    img_color_restored = np.zeros((img_background.shape[0],img_background.shape[1],3), dtype=np.uint8)
    
    img_color_restored[:,:,0] = img_background 
    img_color_restored[:,:,1] = img_background 
    img_color_restored[:,:,2] = img_background 

    # find the points in the colored crop image which are non zero
    pnts = np.transpose(np.asarray(np.nonzero(img_color)))

    # loop over the non zero indices of the crop and appropriately,
    # set the color coordinates in absolute pixels in restored image
    
    for row, col, color in pnts:
        abs_row = x0 + row
        abs_col = y0 + col
        
        img_color_restored[abs_row][abs_col] = img_color[row][col]
        
    return img_color_restored

def processROI(pulse_index, iter_index, exp_index, img, **kwargs):
    """
        0. Zoom out from the center of the image in stages
        
        1. For each stage perform modal analysis for histogram
    """
    
    # get the global dictionary keys
    isWidget = dict_ROI['isWidget']
    
    if isWidget:
        # get the coordinates from the keyword arguments
        coords = kwargs.get('coordinates')

        if coords:
            # interchange rows and columns to reflect coordinate transpose of openCV and numpy
            windowROI = np.array([coords[1][0], coords[0][0], coords[1][2], coords[0][2]])
            windowROI = windowROI.astype(int)
            
            if deep_debug: print(f"WindowROI: {windowROI}")
            
        else:
            # set windowROI to the complete image
            windowROI = np.array([0, 0, img.shape[0], img.shape[1]], dtype=np.int)

    else: 
        #find optimum ROI
        windowROI = findAdaptiveROI(pulse_index, iter_index, exp_index, img) 
        

    return windowROI
    
    
    