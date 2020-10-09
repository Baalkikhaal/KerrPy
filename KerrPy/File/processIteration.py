# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 16:58:34 2020

@author: fubar
"""
import os, os.path
import numpy as np

from globalVariables import debug, dict_image

from KerrPy.File.loadFilePaths import fits_root

from KerrPy.File.processPulse import processPulse

def saveIteration(iter_index, exp_index, iteration):
    """
        Take iter_index, exp_index and iteration array
        as inputs and
        save as numpy array file at level 2 of fits    
    """
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    if debug: print(f"        L2 saveIteration() started at E:{exp_index} I:{iter_index}")
    
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
    
def processIteration(iter_index, exp_index, iter_dir, **kwargs):
    """
        LEVEL 2
        0. Enter the iteration directory
        1. Scan through the pulses
        2. if optional keyword argument `list_counters` is passed,
            0. append the number of images to shape of space.
            1. it is passed onto processPulse()
                for counting the images.
        3. Return the pulses
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
    
    # number of iterations; use global dict_image to find the image files
    n_pulse = dict_image['nImages']
    files = os.listdir()
    files.sort()
    images = files[0:n_pulse]   # ignore initial and final saturation image
    
    if list_counters := kwargs.get('list_counters'):

        # append the number of images to list_space_shape
        list_counters[0][2] = n_pulse

    for i in np.arange(n_pulse):
        pulse_index = i
        img_file = images[i]
        pulse = processPulse(pulse_index, iter_index, exp_index, img_file, **kwargs)
        iteration.append(pulse)

    #save the iteration to file    
    saveIteration(iter_index, exp_index, iteration)
        
    #Restore the path
    os.chdir(cur_path)

    return iteration