# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 16:58:34 2020

@author: fubar
"""
import os, os.path
import numpy as np

from globalVariables import debug, proc_dir, fits_folder, samplename, nImages

from processPulse import processPulse

def saveIteration(iter_index, exp_index, iteration, parent_dir_abs):
    """
        Take iter_index, exp_index and iteration array
        as inputs and
        save as numpy array file at level 2 of fits    
    """
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    if debug: print(f"        L2 saveIteration() started at E:{exp_index} I:{iter_index}")
    
    proc_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, proc_dir))
    if not os.path.isdir(proc_dir_abs): os.mkdir(proc_dir_abs)
    
    fits_dir_abs = os.path.abspath(os.path.join(proc_dir_abs,fits_folder))
    fits_root = os.path.abspath(os.path.join(fits_dir_abs,samplename))
    
    #change to Fits root folder (LEVLE 0)
    os.chdir(fits_root)

    # folder for current experiment; create it if not yet
    fits_exp_folder = f"Experiment_{exp_index}"
    if not os.path.isdir(fits_exp_folder): os.mkdir(fits_exp_folder)
    
    #change to Fits experiment folder (LEVLEL 1)
    os.chdir(fits_exp_folder)

    # folder for current iteration; create it if not yet
    fits_iter_folder = f"Experiment_{exp_index}_Iteration_{iter_index}"
    if not os.path.isdir(fits_iter_folder): os.mkdir(fits_iter_folder)

    #change to Fits iteration folder (LEVEL 2)
    os.chdir(fits_iter_folder)
    
    #save the iteration
    fits_iter_file = f"{fits_iter_folder}.npy"
    np.save(fits_iter_file, np.array(iteration))

    #Restore the path
    os.chdir(cur_path)
    
def processIteration(iter_index, exp_index, iter_dir, parent_dir_abs):
    """
        LEVEL 2
        0. Enter the iteration directory
        1. Scan through the pulses
        2. Return the pulses
    """
    

    
    if debug: print(f"        L2 processIteration() started at E:{exp_index} I:{iter_index}")
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
        
    # enter the iteration directory Level 2
    os.chdir(iter_dir)
    
    # initialize a list for saving iteration
    iteration = []
    
    # an iteration contains pulses
    # so loop the pulses in the iterations
    
    # number of iterations; use global nImages to simplify looping
    n_pulse = nImages
    files = os.listdir()
    files.sort()
    images = files[0:n_pulse]   # ignore initial and final saturation image

    for i in np.arange(n_pulse):
        pulse_index = i
        img_file = images[i]
        pulse = processPulse(pulse_index, iter_index, exp_index, img_file, parent_dir_abs)
        iteration.append(pulse)

    #save the iteration to file    
    saveIteration(iter_index, exp_index, iteration, parent_dir_abs)
        
    #Restore the path
    os.chdir(cur_path)

    return iteration