# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 17:48:40 2020

@author: fubar
"""

import os, os.path
import numpy as np

from globalVariables import debug, proc_dir, fits_folder, samplename, save_exps_file, save_controls_file

def loadFits(**kwargs):
    """
    Load the fits and the controls into memory
    
    In case keyword argument subspace is given,
    return the subspace and the associated controls
    """
    subspace_exps = kwargs.get('subspace_exps', [])
    
    if debug: print(f"subspace_exps: {subspace_exps}")
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    controls_file = os.path.abspath(os.path.join(cur_path, proc_dir, fits_folder, samplename, save_controls_file))
    
    space_file = os.path.abspath(os.path.join(cur_path, proc_dir, fits_folder, samplename, save_exps_file))
    
    space = np.load(space_file)
    
    controls = np.load(controls_file, allow_pickle=True)
    
    #Restore the path to the image path
    os.chdir(cur_path)
    
    if subspace_exps == []:
        
        return controls, space
    else:
        
        return controls[:][subspace_exps], space[subspace_exps]

if __name__ == "__main__":
    
    controls, space = loadFits()