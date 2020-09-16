# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:00:18 2020

@author: fubar
"""
import os, os.path
import numpy as np

from globalVariables import debug, proc_dir, fits_folder, samplename

from KerrPy.File.processIteration import processIteration


def saveExperiment(exp_index, experiment, parent_dir_abs):
    """
        Take exp_index and experiment list as input and save
        as numpy array file at level 1 of fits    
    """
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    if debug: print(f"    L1 saveExperiment() started at E: {exp_index}")
    
    proc_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, proc_dir))
    if not os.path.isdir(proc_dir_abs): os.mkdir(proc_dir_abs)
    
    fits_dir_abs = os.path.abspath(os.path.join(proc_dir_abs,fits_folder))
    fits_root = os.path.abspath(os.path.join(fits_dir_abs,samplename))
    
    #change to Fits root folder (LEVLEL 0)
    os.chdir(fits_root)
    
    #folder for current experiment; create it if not yet
    fits_exp_folder = f"Experiment_{exp_index}"
    if not os.path.isdir(fits_exp_folder): os.mkdir(fits_exp_folder)
    
    #change to Fits experiment folder (LEVEL 1)
    os.chdir(fits_exp_folder)
    
    #save the experiment
    fits_exp_file = f"{fits_exp_folder}.npy"
    np.save(fits_exp_file, np.array(experiment))

    #Restore the path
    os.chdir(cur_path)


def processExperiment(exp_index, exp_dir, parent_dir_abs):
    """
        LEVEL 1
        0. Enter the experiment directory
        1. Scan through the iterations
        2. Return the experiments
    """
    
    if debug: print(f"    L1 processExperiment() started at E: {exp_index}")
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    # enter the experiment directory Level 1
    os.chdir(exp_dir)
    
    
    # initialize a list for saving experiment
    experiment = []
    
    # an experiment contains iterations
    # so loop the iteration directories in the experiment
    files = os.listdir()
    iter_dirs = [each for each in files if os.path.isdir(each)]
    # sort the directories by name
    iter_dirs.sort()

    # number of iterations
    n_iter = len(iter_dirs)
    
    for i in np.arange(n_iter):
        iter_index = i
        iter_dir = iter_dirs[i]
        iteration = processIteration(iter_index, exp_index, iter_dir, parent_dir_abs)
        experiment.append(iteration)
        
    #save the iterations to file    
    saveExperiment(exp_index, experiment, parent_dir_abs)
        
    #Restore the path
    os.chdir(cur_path)

    return experiment
    