# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 19:40:20 2020

@author: fubar
"""

import os, os.path

from KerrPy.Figure.processFigures import processFigures


#disable interactive plotting for this script.
# need to reload matploltib package
import matplotlib as mpl
mpl.style.use('myMatplotlibStylesheet.mplstyle')
mpl.rcParams['text.usetex'] =  True
mpl.use('pdf', force=True)  # set noninteractive backend 'pdf'


parent_dir_abs = os.path.abspath(os.curdir)


#prompt user to input the experiment indices to form a subspace

print("Please enter the experiment indices to form a subspace")

str_exps = input(prompt="Format: 0 2 3 (Press Enter in the end)\n")

# list of subspace experiment indices
subspace_exps = [int(each) for each in str_exps.split()]


controls, filtered_space = processFigures(parent_dir_abs, subspace_exps = subspace_exps)