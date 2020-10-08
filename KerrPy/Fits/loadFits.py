# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 17:48:40 2020

@author: fubar
"""

import numpy as np

from globalVariables import debug

from KerrPy.File.loadFilePaths import controls_filepath, space_filepath

def loadFits(**kwargs):
    """
    Load the fits and the controls into memory
    
    In case keyword argument subspace is given,
    return the subspace and the associated controls
    """
    subspace_exps = kwargs.get('subspace_exps', [])
    
    if debug: print(f"subspace_exps: {subspace_exps}")
    
    space = np.load(space_filepath)
    
    controls = np.load(controls_filepath, allow_pickle=True)
    
    if subspace_exps == []:
        
        return controls, space
    else:
        
        return controls[:][subspace_exps], space[subspace_exps]

if __name__ == "__main__":
    
    controls, space = loadFits()