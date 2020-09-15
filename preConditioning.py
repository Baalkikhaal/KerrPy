# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:49:59 2020

@author: fubar
"""
import os, os.path

from globalVariables import proc_dir, imgs_folder, fits_folder, samplename


def preConditioning():
    """ 
        0. Create the image folder and fits folder accordingly
        1. Return the parent dir absolute path
    """
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    parent_dir_abs = cur_path    
    
    proc_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, proc_dir))
    if not os.path.isdir(proc_dir_abs): os.mkdir(proc_dir_abs)

    
    img_dir_abs = os.path.abspath(os.path.join(proc_dir_abs, imgs_folder))
    if not os.path.isdir(img_dir_abs): os.mkdir(img_dir_abs)
    fits_dir_abs = os.path.abspath(os.path.join(proc_dir_abs, fits_folder))
    if not os.path.isdir(fits_dir_abs): os.mkdir(fits_dir_abs)
    
    img_root = os.path.abspath(os.path.join(img_dir_abs,samplename))
    if not os.path.isdir(img_root): os.mkdir(img_root)
    fits_root = os.path.abspath(os.path.join(fits_dir_abs,samplename))
    if not os.path.isdir(fits_root): os.mkdir(fits_root)
    
    
    return parent_dir_abs
    #pre-conditioning is completed!
