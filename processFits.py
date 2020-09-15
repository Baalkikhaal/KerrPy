# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:19:27 2020

@author: fubar
"""

from preConditioning import preConditioning
from processControls import processControls
from processSpace import processSpace


def processFits():
    """ 0. Do pre-conditioning for saving the fits.
        1. Get the controls from .csv sequence file.
        2. process the experiment space."""
    

    
    parent_dir_abs = preConditioning()
    
    controls = processControls(parent_dir_abs)

    space = processSpace(controls, parent_dir_abs)

    return space

if __name__ == '__main__':
    processFits()