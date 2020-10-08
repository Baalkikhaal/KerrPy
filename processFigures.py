# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:24:00 2020

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



controls, filtered_space = processFigures()

