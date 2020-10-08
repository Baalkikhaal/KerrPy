# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:03:48 2020

@author: fubar
"""

import os, os.path
import numpy as np

from globalVariables import debug, seq_file

from KerrPy.File.loadFilePaths import controls_filepath

def getControls():
    """
        Controls are Hx, Hy, dt (in Oe, Oe, ms)
        
        Discard the remaining controls for now
        
        To avoid np.loadtxt() error when only one experiment,
        use flag ndmin = 2
        Reference : https://stackoverflow.com/questions/13544639/how-to-check-if-an-array-is-2d
        
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

def saveControls(controls):
    """
        Take controls array as input and save
        as numpy array file at level 0 of fits
    """
    
    if debug: print("L0 saveControls() started")
    
    # save the controls
    np.save(controls_filepath, np.array(controls, dtype=object))
    

def processControls():
    """
        0. Scan through the sequence file and get the controls
        1. Save the controls to global {save_controls_file} file

    """

    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)

    if debug: print("L0 processControls() started")

    controls = getControls()
    
    # save the controls
    saveControls(controls)
    
    #Restore the path
    os.chdir(cur_path)
    
    return controls

if __name__ == '__main__':
    test_seq_file = 'Testing/testSequence.csv'
    
    controls = getControls()
    