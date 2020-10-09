# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:17:32 2020

@author: fubar
"""


#######################
##training parameters##
#######################
dict_image = {
                'nucleation_down': True,  # False for nucleation up
                'nImages': 9              # number of pulses in an iteration
              }

dict_ROI = {
    'isWidget'      : False,         # flag for widget based ROI selection 
    # if True the isAdaptive flag and its associates are ignored
    # if False check for isAdaptive flag
    'isAdaptive'    : False,        # flag for adaptive ROI
    # if True the following keys will be used.
        'center'    : (511,672),    #center of the object to be identified
        'aspr'      : 2/3,          # x_width/y_width for zooming out ROI.
        'seq'       : [100,200,300,400,500,600,700,800,900,1000]
                                    # sequence for zoom out
    # if False, then the last ROI of the above sequence
    # is taken as ROI 
            }

dict_openCV =   {
    'kernel_gaussian'   : 5,    # kernel for blurring !should be ODD
                                # typical values 3, 5, 7
    'kernel_median'     : 5,    # kernel for median filtering ! should be ODD
                                # typical values 3, 5, 7
    'kernel_speckless'  : 9,    # kernel for opening and closing! should be ODD
                                # typical values 7, 9, 11
    'speckless_second'  : True,
    'kernel_speckless_second' : 15  # second level opening and closing
                                    # ! should be ODD
                }

dict_fit =  {
    'max_norm_err_sq'   : 20.0, # maximum allowed normalized error
                                # allowed for a point to be an inlier
    'min_confidence'    : 40.0  # minimum confidence to OK a fit
            }                            


##########################
##file handling parameters
##########################.
raw_dir = r'DataSimplified'
proc_dir = r'PostProcessing'

seq_file = r'testSequence.csv'
samplename = r'DataSimplifiedCustomROI'

#########################
######flag parameters###
#########################
debug = True
deep_debug = False
displayImages = False
saveImages = True
saveFigures = True
saveRestoredImages = True

############################
#######matplotlib parameters
############################
mpl_stylesheet = r'myMatplotlibStylesheet.mplstyle'
use_tex = False

############################
##### image parameters #####
############################
image_shape= tuple([1066, 1344])