# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 18:22:23 2020

@author: fubar
"""

import numpy as np

from globalVariables import min_confidence

def filterSpace(space):
    """
        Compare the confidence of fits across the space
        with global min_confidence
        Define badfits as ones with confidence less than
        minimum confidence and extract the indices for
        such pulses.
        
        Set all the pulse parameters corresponding to badfits
        to np.nan
        
        Return the filtered space
    """
    confidence = space[:,:,:,0]
    
    badfits = confidence < min_confidence
    badfits_indices = np.nonzero(badfits)
    
    # ALERT! create a hard copy of space to avoid manipulating it.
    filtered_space = space.copy()
    
    # set the pulse parameters i.e. confidence, a, b, x_c, y_c, o to np.nan
    filtered_space[badfits_indices] = np.nan
    
    return filtered_space
