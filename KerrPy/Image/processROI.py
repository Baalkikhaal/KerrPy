# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:02:03 2020

@author: fubar
"""

import os, os.path
import numpy as np

from globalVariables import debug, proc_dir, imgs_folder, samplename
from globalVariables import center_ROI, aspr_ROI, adaptive_ROI_seq
from globalVariables import displayImages, saveImages

from KerrPy.Image.routinesOpenCV import processImage

from KerrPy.Statistics.routinesSciPy import modalAnalysis

from KerrPy.Figure.routinesMatplotLib import displayImage, plotROIanalysis


def saveROIAnalysis(pulse_index, iter_index, exp_index, plot_ROI_analysis_file, fig_ROI_analysis, parent_dir_abs):
    """
        Take pulse_index, iter_index, exp_index, fig_mod_analysis as inputs
        and save as plots of ROI analysis in ROI_analysis folder
    """
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    if debug: print(f"                L4 saveROIAnalysis() started at E:{exp_index} I:{iter_index} P:{pulse_index}")

    proc_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, proc_dir))
    if not os.path.isdir(proc_dir_abs): os.mkdir(proc_dir_abs)
    
    img_dir_abs = os.path.abspath(os.path.join(proc_dir_abs,imgs_folder))
    img_root = os.path.abspath(os.path.join(img_dir_abs,samplename))
    if not os.path.isdir(img_root): os.mkdir(img_root)
    
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
    
    
    
       

def saveModalAnalysis(pulse_index, iter_index, exp_index, plot_modal_analysis_file, fig_mod_analysis, parent_dir_abs):
    """
        Take pulse_index, iter_index, exp_index, fig_mod_analysis as inputs
        and save as numpy array file at level 3 of fits in modal analysis folder
    """
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    if debug: print(f"                L4 saveModalAnalysis() started at E:{exp_index} I:{iter_index} P:{pulse_index}")

    proc_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, proc_dir))
    if not os.path.isdir(proc_dir_abs): os.mkdir(proc_dir_abs)
    
    img_dir_abs = os.path.abspath(os.path.join(proc_dir_abs,imgs_folder))
    if not os.path.isdir(img_dir_abs): os.mkdir(img_dir_abs)

    img_root = os.path.abspath(os.path.join(img_dir_abs,samplename))
    if not os.path.isdir(img_root): os.mkdir(img_root)
    
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
    
    
def findAdaptiveROI(pulse_index, iter_index, exp_index, img, parent_dir_abs):
    """Using Kernel Density Estimate analysis, relative amount of foreground and background regions within a given ROI can be estimated.
    If the roi is varied across the viewing area, an optimum relative strength can be found when the foreground and background area are nearly equal. This sets the value of adaptive threshold.
    Since we know certain properties about our object of interest like the approximate center of object and the aspect ratio of object area, the ROI can be set accordingly"""

    array_x_ROI     =   np.array(adaptive_ROI_seq)
    array_y_ROI     =   (array_x_ROI*aspr_ROI).astype(int)
    n           =   array_x_ROI.size
    optimum_x_ROI    =  0
    optimum_y_ROI   =   0
    #set the array for relative strengths and maxima positions for the unimodal or bimodal distributions.
    array_rel_strength  =   np.zeros(n)
    array_maximum       =   np.zeros((n,2))   
    for i in np.arange(n):
        
        # custom ROI        
        ROI = [array_x_ROI[i], array_y_ROI[i]]
        
        # process the image
        img_med = processImage(img, ROI)

        img_background = np.zeros([1022,1344], dtype = np.uint8)
        
        img_restored = restoreROI(img_med, img_background)
        
        maximum,rel_strength, fig_mod_analysis    =   modalAnalysis(img_med, img_restored, ROI)    #strength is zero if distribution is unimodal and close to zero if the foreground is very small compared to background or vice versa
        
        if displayImages:
            if saveImages:
                plot_modal_analysis_file = f"ModalAnalysis_xROI = {ROI[0]} yROI = {ROI[1]}"
                saveModalAnalysis(pulse_index, iter_index, exp_index, plot_modal_analysis_file, fig_mod_analysis, parent_dir_abs)
        array_rel_strength[i]   =   rel_strength   
        array_maximum[i]        =   maximum
    
    if displayImages:
        label = "xROI limited"
        fig_ROI_analysis = plotROIanalysis(array_y_ROI, array_rel_strength, label)
        
        if saveImages:
            plot_ROI_analysis_file = "ROIAnalysis_xROIlimited"
            saveROIAnalysis(pulse_index, iter_index, exp_index, plot_ROI_analysis_file, fig_ROI_analysis, parent_dir_abs)

    #if all are unimodal distributions, then there either is no object to be found or object is beyond the ROI. This means that we need to check for bigger ROIs with progressive increase in y axis width
    max_rel_strength = np.max(array_rel_strength)
    
    if debug: print('{}{:.2f}'.format('                max_rel_strength: ', max_rel_strength))
    
    #find the optimum ROI from maximum of the relative strength vs ROI variation
    optimum_x_ROI = array_x_ROI[array_rel_strength.argsort()[-1]]
    optimum_y_ROI = array_y_ROI[array_rel_strength.argsort()[-1]]
    optimum_ROI = (optimum_x_ROI,optimum_y_ROI)
    
    if debug: print(f"                Opt ROI: {optimum_ROI}")
    
    return optimum_ROI


def restoreROI(cropimg, img_background):
    ''' Restore the cropped image onto the 1022 x 1344 canvas'''
    ROI = cropimg.shape
    x_width = ROI[0]
    y_width = ROI[1]
    img_restored = img_background
    x0 = center_ROI[0] - x_width//2
    y0 = center_ROI[1] - y_width//2
#    x1 = center_ROI[0] + x_width//2
#    y1 = center_ROI[1] + y_width//2
    x1 = x0 + cropimg.shape[0]
    y1 = y0 + cropimg.shape[1]    
    img_restored[x0:x1, y0:y1] = cropimg
    
    if displayImages:
        title = 'Restored image'
        displayImage(img_restored, title)
    return img_restored

def restoreColorROI(img_color, img_background):
    """
        Restore the colored crop image with ellipse fit
        
        onto the background
    """
    ROI = img_color.shape

    x_width = ROI[0]
    y_width = ROI[1]
    
    img_color_restored = np.zeros((img_background.shape[0],img_background.shape[1],3), dtype=np.uint8)
    
    img_color_restored[:,:,0] = img_background 
    img_color_restored[:,:,1] = img_background 
    img_color_restored[:,:,2] = img_background 

    
    # origin of the ROI in absolute coordinates
    x0 = center_ROI[0] - x_width//2
    y0 = center_ROI[1] - y_width//2
    
    # find the points in the colored crop image which are non zero
    
    pnts = np.transpose(np.asarray(np.nonzero(img_color)))

    # print(f"pnts: {pnts}")
    
    # loop over the non zero indices of the crop and appropriately,
    # set the color coordinates in absolute pixels in restored image
    
    for row, col, color in pnts:
        abs_row = x0 + row
        abs_col = y0 + col
        
        img_color_restored[abs_row][abs_col] = img_color[row][col]
        
    return img_color_restored

def restoreCustomROI(cropimg, img_background, customROI):
    ''' Restore the cropped image onto the 1022 x 1344 canvas'''

    # transpose the coordinates as openCV and numpy coordinate systems are tranpose to each other
    origin_x_customROI = customROI[0]
    origin_y_customROI = customROI[1]
    
    end_x_customROI = customROI[2]
    end_y_customROI = customROI[3]
    
    img_restored = img_background 

    # origin of the ROI in absolute coordinates
    x0 = origin_x_customROI
    y0 = origin_y_customROI

    x1 = end_x_customROI
    y1 = end_y_customROI    
    img_restored[x0:x1, y0:y1] = cropimg
    
    if displayImages:
        title = 'Restored image'
        displayImage(img_restored, title)
    return img_restored

def restoreColorCustomROI(img_color, img_background, customROI):
    """
        Restore the colored crop image with ellipse fit
        
        onto the background
    """
    
    
    # x_width = customROI[2] - customROI[0]
    # y_width = customROI[3] - customROI[1]
    
    # transpose the coordinates as openCV and numpy coordinate systems are tranpose to each other
    origin_x_customROI = customROI[0]
    origin_y_customROI = customROI[1]
    
    img_color_restored = np.zeros((img_background.shape[0],img_background.shape[1],3), dtype=np.uint8)
    
    img_color_restored[:,:,0] = img_background 
    img_color_restored[:,:,1] = img_background 
    img_color_restored[:,:,2] = img_background 

    
    # origin of the ROI in absolute coordinates
    x0 = origin_x_customROI
    y0 = origin_y_customROI
    
    # find the points in the colored crop image which are non zero
    
    pnts = np.transpose(np.asarray(np.nonzero(img_color)))

    # print(f"pnts: {pnts}")
    
    # loop over the non zero indices of the crop and appropriately,
    # set the color coordinates in absolute pixels in restored image
    
    for row, col, color in pnts:
        abs_row = x0 + row
        abs_col = y0 + col
        
        img_color_restored[abs_row][abs_col] = img_color[row][col]
        
    return img_color_restored

def processROI(pulse_index, iter_index, exp_index, img, parent_dir_abs):
    """
        0. Zoom out from the center of the image in stages
        
        1. For each stage perform modal analysis for histogram
    """
    ROI = findAdaptiveROI(pulse_index, iter_index, exp_index, img, parent_dir_abs)
    
    return ROI
    
    
    