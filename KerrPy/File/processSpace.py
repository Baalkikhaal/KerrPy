# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:02:16 2020

@author: fubar
"""

import os, os.path
import numpy as np

from globalVariables import debug, deep_debug

from KerrPy.File.loadFilePaths import raw_dir_abs, space_filepath

from KerrPy.File.processExperiment import processExperiment

def extractControlsFromFolderName(foldername):
    """
        splits the foldername for eg.,
        foldername = 'NUCLEATE-M_IPBHx0.00_Ip-2.20E+1_tp1.20E+3_DUR CYC5'
        
        and extracts Hx = 0, Hz = -22 tp = 120
        
        For a resolution of 0.5 Oe we need precision of 1
        For this use formating '{:.1f}'.format(float)
        Reference : https://pyformat.info/
    """
    
    splits = foldername.split('_')
    
    Hip     =   '{:.1f}'.format(np.float(splits[1][5:]))
    Hop     =   '{:.1f}'.format(np.float(splits[2][2:]))
    tp      =   '{:.1f}'.format(np.float(splits[3][2:]))    
    
    controls = [Hip,Hop,tp]


    if deep_debug : print (f"    extracted controls: {controls}")
    
    return controls

def findExperiment(exp_controls):
    """
    Take the controls of experiment as input,
    Enter the raw directory and
    return the directory corresponding to
    the experiment

    """
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    #change to raw_dir
    os.chdir(raw_dir_abs)#level 1
    
    files = os.listdir()
    dirs = [each for each in files if os.path.isdir(each)]
    
    exp_dir = ''
    for each in dirs:
        controls_extracted = extractControlsFromFolderName(each)
        
        if (controls_extracted == exp_controls[0]):
            exp_dir = each
            break
    if not exp_dir:
        if debug: 
            print(f"FATAL! Experiment {exp_controls} not found!")
    
    #Restore the path
    os.chdir(cur_path)
    
    return exp_dir

def saveSpace(space):
    """
        Created on Sat Sep 12 17:02:16 2020
        
        @author: fubar
        
        Take space list as input and save
        as numpy array file at level 0 of fits
    """
    
    if debug: print("L0 saveSpace() started")
    
    # save the space
    np.save(space_filepath, np.array(space))
    

def processSpace(controls, **kwargs):
    """
        Created on Sat Sep 12 17:02:16 2020

        @author: fubar
        
        0. Takes controls as input.
        
        1. Scan through the experiments in raw dir
        
        2. For this use the controls to cycle through the experiments,
        instead of listing the directories.
        
        3. if optional keyword argument `list_counters` is passed,
            0. append the number of experiments to shape of space
            
            1. it is passed onto processExperiment()
            for counting the images.
        
    """
    
    if debug: print("L0 processSpace() started")
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    # enter the experiment directory Level 1
    os.chdir(raw_dir_abs)

    
    # initialize a list for saving space
    space = []
    
    # a space contains experiments
    n_exp = len(controls)
    
    if list_counters := kwargs.get('list_counters'):
        
        # append the number of experiments to list_space_shape
        list_counters[0][0] = n_exp

    # now loop the experiments in the controls sequence
    for i in np.arange(n_exp):
        exp_index = i
        exp_controls = controls[i]
        
        if debug: print(f"L0 E: {exp_index} = {exp_controls}")
        
        # find the experiment directory corresponding to experiment index

        exp_dir = findExperiment(exp_controls)
        
        experiment = processExperiment(exp_index, exp_dir, **kwargs)
        
        space.append(experiment)
        
    saveSpace(space)
    
    #Restore the path
    os.chdir(cur_path)
    
    return space

if __name__ == '__main__':
    foldername = 'NUCLEATE-M_IPBHx0.00_Ip-1.00E+1_tp6.50E+3_DUR CYC5'
    exp_controls = extractControlsFromFolderName(foldername)
    print(exp_controls)