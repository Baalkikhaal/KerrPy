# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:19:27 2020

@author: fubar
"""

import os, os.path

from KerrPy.File.preConditioning import preConditioning
from KerrPy.File.processControls import processControls
from KerrPy.File.processSpace import processSpace


def processFits():
    """ 0. Do pre-conditioning for saving the fits.
        1. Get the controls from .csv sequence file.
        2. process the experiment space."""
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    parent_dir_abs = preConditioning()
    
    controls = processControls(parent_dir_abs)

    space = processSpace(controls, parent_dir_abs)
    
    #Restore the path to the image path
    os.chdir(cur_path)

    return controls, space

if __name__ == '__main__':
    controls, space = processFits()