# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:19:27 2020

@author: fubar
"""

import os, os.path

from KerrPy.File.processControls import processControls
from KerrPy.File.processSpace import processSpace, processSpaceWithCustomROI


def processData():
    """ 0. Do pre-conditioning for saving the fits.
        1. Get the controls from .csv sequence file.
        2. process the experiment space."""
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    controls = processControls()

    space = processSpace(controls)
    
    #Restore the path to the image path
    os.chdir(cur_path)

    return controls, space

def processDataWithCustomROI(list_counters):
    """ 0. Do pre-conditioning for saving the fits.
        1. Get the controls from .csv sequence file.
        2. process the experiment space."""
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    controls = processControls()

    list_counters = processSpaceWithCustomROI(list_counters, controls)
    
    #Restore the path to the image path
    os.chdir(cur_path)

    # return controls, space

    return list_counters

if __name__ == '__main__':
    controls, space = processData()