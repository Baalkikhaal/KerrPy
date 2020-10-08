# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 19:40:20 2020

@author: fubar

# mpl.use('pdf', force=True)  # set noninteractive backend 'pdf'
# to check the available non gui backends
#   non_gui_backends = mpl.rcsetup.non_interactive_bk
#   print(non_gui_backends)
#   to check gui backends
#   gui_backends = mpl.rcsetup.interactive_bk
#   Reference : https://stackoverflow.com/questions/3285193/how-to-change-backends-in-matplotlib-python
"""

from KerrPy.Figure.processFigures import processFigures


#disable interactive plotting for this script.
# need to reload matploltib package
import matplotlib as mpl

from globalVariables import mpl_stylesheet, use_tex

mpl.style.use(mpl_stylesheet)
mpl.rcParams['text.usetex'] =  use_tex

mpl.use('pdf', force=True)  # set noninteractive backend 'pdf'



#prompt user to input the experiment indices to form a subspace

print("Please enter the experiment indices to form a subspace")

str_exps = input(prompt="Format: 0 2 3 (Press Enter in the end)\n")

# list of subspace experiment indices
subspace_exps = [int(each) for each in str_exps.split()]


controls, filtered_space = processFigures(subspace_exps = subspace_exps)