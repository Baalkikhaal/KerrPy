# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:03:48 2020

@author: fubar
"""

import os, os.path
import numpy as np

from globalVariables import seq_file
from globalVariables import debug, proc_dir, fits_folder, samplename
from globalVariables import save_controls_file


def getControls():
    """
        Controls are Hx, Hy, dt (in Oe, Oe, ms)
        
        Discard the remaining controls for now
        
        To avoid np.loadtxt() error when only one experiment,
        use flag ndmin = 2
        Reference : https://stackoverflow.com/questions/13544639/how-to-check-if-an-array-is-2d
        
        # TODO : NOT working for float fields
                For a resolution of 0.5 Oe we need precision of 1
                For this use formating '{:.1f}'.format(float)
                Reference : https://pyformat.info/
    """
    exp_seq = np.loadtxt(seq_file, delimiter=',',skiprows=1, ndmin= 2 )
    N = exp_seq.shape[0]
    if debug: print("number of experiments is " + str(N))

    #initialize the list for controls
    controls = []
    
    for i in np.arange(N):
        Hip     =   '{:.1f}'.format(exp_seq[i][4])   
        Hop     =   '{:.1f}'.format(exp_seq[i][0])   
        tp      =   '{:.1f}'.format(exp_seq[i][1])
        
        exp_control = [[Hip, Hop, tp], i]
        
        controls.append(exp_control)
    

    if debug: print("Control Knobs for experiments are \n" + str(controls))
    
    return controls

def saveControls(controls, parent_dir_abs):
    """
        Take controls array as input and save
        as numpy array file at level 0 of fits
    """
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    if debug: print("L0 saveControls() started")
    
    proc_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, proc_dir))
    if not os.path.isdir(proc_dir_abs): os.mkdir(proc_dir_abs)
    
    fits_dir_abs = os.path.abspath(os.path.join(proc_dir_abs,fits_folder))
    fits_root = os.path.abspath(os.path.join(fits_dir_abs,samplename))
    
    #change to Fits root folder (LEVLEL 0)
    os.chdir(fits_root)
    
    # save the controls
    controls_filename = save_controls_file
    controls_filepath = os.path.abspath(os.path.join(fits_root,controls_filename))
    np.save(controls_filepath, np.array(controls))
    
    #Restore the path
    os.chdir(cur_path)

def processControls(parent_dir_abs):
    """
        0. Scan through the sequence file and get the controls
        1. Save the controls to global {save_controls_file} file

    """

    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)

    if debug: print("L0 processControls() started")

    controls = getControls()
    
    # save the controls
    saveControls(controls, parent_dir_abs)
    
    #Restore the path
    os.chdir(cur_path)
    
    return controls

if __name__ == '__main__':
    test_seq_file = 'Testing/testSequence.csv'
    
    controls = getControls()
    