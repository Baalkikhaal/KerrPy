# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:40:30 2020

@author: fubar
"""

import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from ScanThroughCrops import RestoreROI

if __name__ == '__main__':
    displayImages = True
    DEBUG = True
    
    center_ROI      =   (511,672)  #center of the object to be identified
    cropimg = cv2.imread('00_Cropped.png',0)

    image_color = cv2.merge((cropimg, cropimg, cropimg))
    
    image_ellipsepath = '09_FitEllipse.png'
    img_ellipse = cv2.imread(image_ellipsepath, 3)
    
    
    mpl.style.use('myMatplotlibStylesheet.mplstyle')

    mpl.rcParams['figure.max_open_warning'] = 40
    
    x_width = 1022
    y_width = 1344
    
    RestoreROI(img_ellipse, image_color, center_ROI, x_width, y_width, displayImages, DEBUG=True)