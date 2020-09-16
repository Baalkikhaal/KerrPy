# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:02:03 2020

@author: fubar
"""

import os, os.path
import numpy as np

from globalVariables import debug, proc_dir, imgs_folder, samplename
from globalVariables import center_ROI, aspr_ROI
from globalVariables import displayImages, saveImages

from routinesOpenCV import processImage

from routinesSciPy import modalAnalysis

from routinesMatplotLib import displayImage, plotROIanalysis


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

    array_x_ROI     =   np.array([100,200,300,400,500,600,700,800,900,1000])
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
        
        img_restored = restoreROI(img_med, img_background, ROI)
        
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
    
    if max_rel_strength < 0.001:
        optimum_x_ROI = 1000
    else:
        #find the optimum ROI from maximum of the relative strength vs ROI variation
        optimum_x_ROI = array_x_ROI[array_rel_strength.argsort()[-1]]
        optimum_y_ROI = array_y_ROI[array_rel_strength.argsort()[-1]]

    #if optimum ROI is less than 1000, then it probably means that the object is not occluded and search for the ROI is completed. If the ROI is not optimized then we can increase the y_width of ROI further keeping the x_width to be constant at 1022
    if optimum_x_ROI == 1000:
        array_y_ROI  = np.array([800,900,1000,1100,1200,1300])
        n           = array_y_ROI.size
        array_x_ROI = np.ones(n,dtype = np.int32)*1022
        #set the array for relative strengths and maxima positions for the unimodal or bimodal distributions.
        array_rel_strength  =   np.zeros(n)
        array_maximum       =   np.zeros((n,2))   
        for i in np.arange(n):
 
            # custom ROI
            ROI = [array_x_ROI[i], array_y_ROI[i]]
        
            # process the image
            img_med = processImage(img, ROI)
        
            img_background = np.zeros([1022,1344], dtype = np.uint8)
            img_restored = restoreROI(img_med, img_background, ROI)
            maximum, rel_strength, fig_mod_analysis    =   modalAnalysis(img_med, img_restored, ROI)    #strength is zero if distribution is unimodal and close to zero if the foreground is very small compared to background or vice versa
            if displayImages:
                if saveImages:
                    plot_modal_analysis_file = f"ModalAnalysis_xROI = {ROI[0]} yROI = {ROI[1]}"
                    saveModalAnalysis(pulse_index, iter_index, exp_index, plot_modal_analysis_file, fig_mod_analysis, parent_dir_abs)
            array_rel_strength[i]   =   rel_strength   
            array_maximum[i]        =   maximum
        if displayImages:
            label = "yROI limited"
            fig_ROI_analysis = plotROIanalysis(array_y_ROI, array_rel_strength, label)

            if saveImages:
                plot_ROI_analysis_file = "ROIAnalysis_yROIlimited"
                saveROIAnalysis(pulse_index, iter_index, exp_index, plot_ROI_analysis_file, fig_ROI_analysis, parent_dir_abs)

        max_rel_strength = np.max(array_rel_strength)
        if max_rel_strength == 0:
            optimum_x_ROI = 0
            optimum_y_ROI = 0
            if debug: print("                Discard image!")
        #find the optimum ROI from maximum of the relative strength vs ROI variation
        optimum_x_ROI = array_x_ROI[array_rel_strength.argsort()[-1]]
        optimum_y_ROI = array_y_ROI[array_rel_strength.argsort()[-1]]
        if optimum_y_ROI == 1300:
            #so the whole image needs to be used for further processing
            optimum_x_ROI = 1022
            optimum_y_ROI = 1344
        #proceed with further processing with optimum ROI
    optimum_ROI = (optimum_x_ROI,optimum_y_ROI)
    if debug: print(f"                Opt ROI: {optimum_ROI}")
    return optimum_ROI


def restoreROI(cropimg, img_background, ROI):
    ''' Restore the cropped image onto the 1022 x 1344 canvas'''
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


def processROI(pulse_index, iter_index, exp_index, img, parent_dir_abs):
    """
        0. Zoom out from the center of the image in stages
        
        1. For each stage perform modal analysis for histogram
    """
    ROI = findAdaptiveROI(pulse_index, iter_index, exp_index, img, parent_dir_abs)
    
    return ROI
    
    
    