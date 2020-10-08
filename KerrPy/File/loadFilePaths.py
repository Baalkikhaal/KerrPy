# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:15:29 2020

@author: fubar
"""

import os, os.path

from globalVariables import debug, raw_dir, proc_dir, samplename

from globalVariables import mpl_stylesheet, __file__


#### global read only #######

imgs_folder = r'Images'
fits_folder = r'Fits'
restored_folder = r'Restored'
figs_folder = r'Figures'
vel_folder = r'Velocity'

space_filename = 'space.npy'
controls_filename = 'controls.npy'

######### Post Processing folders and files ###################

# From the `__file__` attribute of globalVariables, let us find the 
# parent directory. We assume globalVariables is in the parent directory
parent_dir_abs = os.path.dirname(__file__)

if debug: print(f'parent_dir_abs: {parent_dir_abs}')

raw_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, raw_dir))
if not os.path.isdir(raw_dir_abs): os.mkdir(raw_dir_abs)

proc_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, proc_dir))
if not os.path.isdir(proc_dir_abs): os.mkdir(proc_dir_abs)

fits_dir_abs = os.path.abspath(os.path.join(proc_dir_abs,fits_folder))
if not os.path.isdir(fits_dir_abs): os.mkdir(fits_dir_abs)

fits_root = os.path.abspath(os.path.join(fits_dir_abs,samplename))
if not os.path.isdir(fits_root): os.mkdir(fits_root)

space_filepath = os.path.abspath(os.path.join(fits_root,space_filename))

controls_filepath = os.path.abspath(os.path.join(fits_root, controls_filename))

######images folders#########

# image fits folder root
img_dir_abs = os.path.abspath(os.path.join(proc_dir_abs, imgs_folder))
if not os.path.isdir(img_dir_abs): os.mkdir(img_dir_abs)

img_root = os.path.abspath(os.path.join(img_dir_abs,samplename))
if not os.path.isdir(img_root): os.mkdir(img_root)

# restored image fits folder root
restored_dir_abs = os.path.abspath(os.path.join(proc_dir_abs, restored_folder))
if not os.path.isdir(restored_dir_abs): os.mkdir(restored_dir_abs)

restored_root = os.path.abspath(os.path.join(restored_dir_abs,samplename))
if not os.path.isdir(restored_root): os.mkdir(restored_root)


######### figures folder ########

figs_dir_abs = os.path.abspath(os.path.join(proc_dir_abs, figs_folder))
if not os.path.isdir(figs_dir_abs): os.mkdir(figs_dir_abs)

figs_root = os.path.abspath(os.path.join(figs_dir_abs,samplename))
if not os.path.isdir(figs_root): os.mkdir(figs_root)

######## velocity folder #######

vel_dir_abs = os.path.abspath(os.path.join(proc_dir_abs, vel_folder))
if not os.path.isdir(vel_dir_abs): os.mkdir(vel_dir_abs)

vel_root = os.path.abspath(os.path.join(vel_dir_abs,samplename))
if not os.path.isdir(vel_root): os.mkdir(vel_root)

######## matplotlib stylesheet file #######

mpl_stylesheet_file = os.path.abspath(os.path.join(parent_dir_abs, mpl_stylesheet))

