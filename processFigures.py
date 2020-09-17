# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:24:00 2020

@author: fubar
"""

import os, os.path

from KerrPy.Figure.processFigures import processFigures

parent_dir_abs = os.path.abspath(os.curdir)

controls, filtered_space = processFigures(parent_dir_abs)

