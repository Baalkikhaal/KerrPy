# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:17:32 2020

@author: fubar
"""


#######################
##training parameters##
#######################
nucleation_down =   1       # 0 for nucleation up
nImages         =   9       # number of pulses in an iteration
center_ROI      =   (511,672)  #center of the object to be identified
adaptive_ROI    =   False    # flag for adaptive ROI
aspr_ROI        =   2/3     # x_width/y_width for zooming out ROI.
                            # will be used if adaptive_ROI is True
adaptive_ROI_seq =  [100,200,300,400,500,600,700,800,900,1000] 
                            # sequence for zoom out
custom_ROI = True
kernel_gaussian =   5       # kernel for blurring !should be ODD
                            # typical values 3, 5, 7
kernel_median   =   5       # kernel for median filtering ! should be ODD
                            # typical values 3, 5, 7
kernel_speckless=   9       # kernel for opening and closing! should be ODD
                            # typical values 7, 9, 11
speckless_second = True
kernel_speckless_second = 15    # second level opening and closing
                            # ! should be ODD
max_norm_err_sq =   20.0    # maximum allowed normalized error
                            # allowed for a point to be an inlier
min_confidence  =   40.0    # minimum confidence for fit to be considered OK


##########################
##file handling parameters
##########################.
raw_dir = r'DataInFocus'
proc_dir = r'PostProcessing'
imgs_folder = r'Images'
fits_folder = r'Fits'
restored_folder = r'Restored'
figs_folder = r'Figures'
vel_folder = r'Velocity'
seq_file = r'testSequence.csv'
samplename = r'DataInFocus'

save_exps_file = 'space.npy'
save_controls_file = 'controls.npy'


#########################
######debug parameters###
#########################
debug = True
deep_debug = True
displayImages = False
saveImages = True
saveFigures = True
saveRestoredImages = True
