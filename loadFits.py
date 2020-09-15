# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 17:48:40 2020

@author: fubar
"""

import os, os.path
import numpy as np

from globalVariables import proc_dir, fits_folder, samplename, save_exps_file, save_controls_file

#Store the current location before relocating
cur_path = os.path.abspath(os.curdir)

controls_file = os.path.abspath(os.path.join(cur_path, proc_dir, fits_folder, samplename, save_controls_file))

space_file = os.path.abspath(os.path.join(cur_path, proc_dir, fits_folder, samplename, save_exps_file))

space = np.load(space_file)

controls = np.load(controls_file)
