# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 18:39:00 2020

@author: fubar
"""

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from fitEllipse1 import FitEllipse_LeastSquares, OverlayRANSACFit, ConfidenceInFit

mpl.style.use('myMatplotlibStylesheet.mplstyle')
mpl.rcParams['figure.figsize'] = 4.0,2.0
mpl.rcParams['figure.subplot.bottom'] = 0.1
mpl.rcParams['figure.subplot.left'] = 0.1
mpl.rcParams['figure.subplot.right'] = 0.90
mpl.rcParams['figure.subplot.top'] = 0.90

if __name__ == '__main__':
    unittest = True
    DEBUG= True
    max_norm_err_sq =   20.0    # maximum allowed normalized error allowed for a point to be an inlier

#    img_edges = cv2.imread('06_CannyEdges.png', 0)
    
    
    #load the Canny edges
    edges = np.load('08_CannyEdges.npy')
    
    #load the median filtered image
    img_med = cv2.imread('05_MedianFiltered.png',0)
    
    #create a color image
#    img_color = cv2.merge((edges, edges, edges))
    img_color = cv2.merge((img_med, img_med, img_med))
    pnts = np.transpose(np.nonzero(edges))

    ellipse = FitEllipse_LeastSquares(pnts)
    perc_inliers, inlier_pnts, norm_err = ConfidenceInFit (pnts, ellipse, max_norm_err_sq, DEBUG)
    
    OverlayRANSACFit(img_color, pnts, inlier_pnts, ellipse)
