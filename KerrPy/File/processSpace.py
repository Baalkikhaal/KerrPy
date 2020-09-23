# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:02:16 2020

@author: fubar
"""

import os, os.path
import numpy as np

from globalVariables import debug, deep_debug, proc_dir, fits_folder, samplename
from globalVariables import save_exps_file
from globalVariables import raw_dir

from KerrPy.File.processExperiment import processExperiment, processExperimentWithWindow

def extractControlsFromFolderName(foldername):
    """
        splits the foldername for eg.,
        foldername = 'NUCLEATE-M_IPBHx0.00_Ip-2.20E+1_tp1.20E+3_DUR CYC5'
        
        and extracts Hx = 0, Hz = -22 tp = 120
        
        # TODO : NOT working for float fields!
                For a resolution of 0.5 Oe we need precision of 1
                For this use formating '{:.1f}'.format(float)
                Reference : https://pyformat.info/
    """
    
    splits = foldername.split('_')
    
    Hip     =   '{:.1f}'.format(np.float(splits[1][5:]))
    Hop     =   '{:.1f}'.format(np.float(splits[2][2:]))
    tp      =   '{:.1f}'.format(np.float(splits[3][2:]))    
    
#    controls = np.array([Hip,Hop,tp],dtype=np.int32)
    controls = [Hip,Hop,tp]


    if deep_debug : print (f"    extracted controls: {controls}")
    
    return controls

def findExperiment(exp_controls, parent_dir_abs):
    """
    Take the controls of experiment as input,
    Enter the raw directory and
    return the directory corresponding to
    the experiment

    """
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    raw_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, raw_dir))
    if not os.path.isdir(raw_dir_abs): os.mkdir(raw_dir_abs)
    
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

def saveSpace(space, parent_dir_abs):
    """
        Created on Sat Sep 12 17:02:16 2020
        
        @author: fubar
        
        Take space list as input and save
        as numpy array file at level 0 of fits
    """
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    if debug: print("L0 saveSpace() started")
    
    proc_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, proc_dir))
    if not os.path.isdir(proc_dir_abs): os.mkdir(proc_dir_abs)
    
    fits_dir_abs = os.path.abspath(os.path.join(proc_dir_abs,fits_folder))
    fits_root = os.path.abspath(os.path.join(fits_dir_abs,samplename))
    
    #change to Fits root folder (LEVLEL 0)
    os.chdir(fits_root)
    
    # save the space
    space_filename = save_exps_file
    space_filepath = os.path.abspath(os.path.join(fits_root,space_filename))
    np.save(space_filepath, np.array(space))
    
    #Restore the path
    os.chdir(cur_path)
    

def processSpace(controls, parent_dir_abs):
    """
        Created on Sat Sep 12 17:02:16 2020

        @author: fubar
        
        Takes controls as input.
        
        Scan through the experiments in raw dir
        
        For this use the controls to cycle through the experiments,
        instead of listing the directories.
        
        
    """
    
    if debug: print("L0 processSpace() started")
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    # enter the experiment directory Level 1
    os.chdir(raw_dir)

    
    # initialize a list for saving space
    space = []
    
    # a space contains experiments
    # so loop the experiments in the controls sequence
    n_exp = len(controls)

    for i in np.arange(n_exp):
        exp_index = i
        exp_controls = controls[i]
        
        if debug: print(f"L0 E: {exp_index} = {exp_controls}")
        
        # find the experiment directory corresponding to experiment index

        exp_dir = findExperiment(exp_controls, parent_dir_abs)
        
        experiment = processExperiment(exp_index, exp_dir, parent_dir_abs)
        
        space.append(experiment)
        
    saveSpace(space, parent_dir_abs)
    
    #Restore the path
    os.chdir(cur_path)
    
    return space


def processSpaceWithWindow(static_ax, dynamic_ax, controls, parent_dir_abs):
    """
        pass in window along with usual processSpace()
    """
    
    if debug: print("L0 processSpace() started")
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    # enter the experiment directory Level 1
    os.chdir(raw_dir)

    
    # initialize a list for saving space
    space = []
    
    # a space contains experiments
    # so loop the experiments in the controls sequence
    n_exp = len(controls)

    for i in np.arange(n_exp):
        exp_index = i
        exp_controls = controls[i]
        
        if debug: print(f"L0 E: {exp_index} = {exp_controls}")
        
        # find the experiment directory corresponding to experiment index

        exp_dir = findExperiment(exp_controls, parent_dir_abs)
        
        experiment = processExperimentWithWindow(static_ax, dynamic_ax, exp_index, exp_dir, parent_dir_abs)
        
        space.append(experiment)
        
    saveSpace(space, parent_dir_abs)
    
    #Restore the path
    os.chdir(cur_path)
    
    return space


if __name__ == '__main__':
    foldername = 'NUCLEATE-M_IPBHx0.00_Ip-1.00E+1_tp6.50E+3_DUR CYC5'
    exp_controls = extractControlsFromFolderName(foldername)
    print(exp_controls)