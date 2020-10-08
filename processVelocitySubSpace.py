# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 00:42:39 2020

@author: fubar
"""

from KerrPy.Fits.processVelocity import processVelocity


#prompt user to input the experiment indices to form a subspace

print("Please enter the experiment indices to form a subspace")

str_exps = input(prompt="Format: 0 2 3 (Press Enter in the end)\n")

# list of subspace experiment indices
subspace_exps = [int(each) for each in str_exps.split()]

velocity_um_per_sec = processVelocity(subspace_exps = subspace_exps)