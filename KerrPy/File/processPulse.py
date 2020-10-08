# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 16:38:26 2020

@author: fubar
"""
import os, os.path
import numpy as np

from globalVariables import debug

from KerrPy.File.loadFilePaths import fits_root

from KerrPy.Image.processOpenCV import processOpenCV

def savePulse(pulse_index, iter_index, exp_index, pulse):
    """
        Take pulse_index, iter_index, exp_index, pulse array as inputs
        and save as numpy array file at level 3 of fits.
    """
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    if debug: print(f"            L3 savePulse() started at E:{exp_index} I:{iter_index} P:{pulse_index}")

    #change to Fits root folder (LEVEL 0)
    os.chdir(fits_root)

    # folder for current experiment; create it if not yet
    fits_exp_folder = f"Experiment_{exp_index}"
    if not os.path.isdir(fits_exp_folder): os.mkdir(fits_exp_folder)
    
    #change to Fits experiment folder (LEVEL 1)
    os.chdir(fits_exp_folder)

    # folder for current iteration; create it if not yet
    fits_iter_folder = f"Experiment_{exp_index}_Iteration_{iter_index}"
    if not os.path.isdir(fits_iter_folder): os.mkdir(fits_iter_folder)

    #change to Fits iteration folder (LEVEL 2)
    os.chdir(fits_iter_folder)

    # folder for current pulse; create it if not yet
    fits_pulse_folder = f"Experiment_{exp_index}_Iteration_{iter_index}_Pulse_{pulse_index}"
    if not os.path.isdir(fits_pulse_folder): os.mkdir(fits_pulse_folder)
    
    #change to Fits pulse folder (LEVEL 3)
    os.chdir(fits_pulse_folder)
    
    #save the pulse
    fits_params_file = f"{fits_pulse_folder}.npy"
    np.save(fits_params_file, np.array(pulse))

    #Restore the path to the image path
    os.chdir(cur_path)

def processPulse(pulse_index, iter_index, exp_index, img_file):
    """
        LEVEL 3
        0. Fit the ellipse to the bubble
        1. Save the fits
    """
    
    if debug: print(f"            L3 processPulse() started at E:{exp_index} I:{iter_index} P:{pulse_index}")

    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    ######################
    #Find the ellipse fit#
    ######################
    pulse = processOpenCV(pulse_index, iter_index, exp_index, img_file)
    
    #save the pulse to file    
    savePulse(pulse_index, iter_index, exp_index, pulse)
    
    #Restore the path
    os.chdir(cur_path)

    
    return pulse

def processPulseWithCustomROI(list_counters, pulse_index, iter_index, exp_index, img_file):
    """
        LEVEL 3
        0. Fit the ellipse to the bubble
        1. Save the fits
    """
    
    if debug: print(f"            L3 processPulse() started at E:{exp_index} I:{iter_index} P:{pulse_index}")

    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    # absolute path to the image
    img_file_abs = os.path.abspath(os.path.join(cur_path, img_file))
    
    ######################
    #Find the ellipse fit#
    ######################
    
    n_params = 6
    
    # append the number of params to list_space_shape
    list_counters[0][3] = n_params
    
    #instead of ellipse fitting, let us append the indices of the pulse, 
    # iteration, experiment to the counters
    
    list_pulse_index = list_counters[1]
    list_iter_index = list_counters[2]
    list_exp_index = list_counters[3]
    list_img_file = list_counters[4]
    
    list_pulse_index.append(pulse_index)
    list_iter_index.append(iter_index)
    list_exp_index.append(exp_index)
    list_img_file.append(img_file_abs)
    
    #Restore the path
    os.chdir(cur_path)

    return list_counters