# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 17:19:06 2020

@author: fubar

"""

import os, os.path

from KerrPy.Fits.processVelocity import processVelocity

parent_dir_abs = os.path.abspath(os.curdir)

velocity_um_per_sec = processVelocity(parent_dir_abs)