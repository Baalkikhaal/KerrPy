# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:57:38 2018

@author: fubar
"""

from scipy import stats
from scipy.signal import argrelextrema

import numpy as np

from globalVariables import debug, deep_debug, displayImages

from KerrPy.Figure.routinesMatplotLib import plotModAnalysis

#x1 = np.array([-7, -5, 1, 4, 5], dtype=np.float)

def my_kde_bandwidth(obj, fac=5):
    """
        We use Scott's Rule for bandwidth estimation
        However to smoothen the Kernel density estimate
        we multiply by a constant factor.
        
        # TODO migrate the factor to globalVariables
    """
    return np.power(obj.n, -1./(obj.d+4)) * fac
#kde1 = stats.gaussian_kde(x1)
#kde2 = stats.gaussian_kde(x1, bw_method='silverman')


def modalAnalysis(image, img_restored, ROI):
    """
        0. Perform univariate modal analysis on the
            histogram of `image`. Kernel density estimation
            is performed using gaussian_kde() from `stats`
            module of scipy package.
            
        1. We assume it is a bimodal distribution.
        
        2. For a given ROI, to find the modes,
            we find the two peaks with maximum intensity.
            
        3. ROI is restored onto `img_restored`
            
        # TODO 
            Sometimes peak estimation is erroneous
            due to the presence of topographic defects
            
            Also with smaller bandwidth, kde has multiple peaks
            However it can be observed that there is a clustering
            of the peaks.
            
            A better algorithm would be to find the number of clusters
            label the clusters according to their expected range.
            
            Select the clusters corresponding to bubble and background
            
            Then perform bimodal fitting over the filtered clusters.

    """
    x_width = ROI[0]
    y_width = ROI[1]
    #plot the histogram of pixels

    hist = np.histogram(image, bins = 255)
    x1 = image.reshape((1,image.shape[0]*image.shape[1]))

    # kde = stats.gaussian_kde(x1)
    kde = stats.gaussian_kde(x1, bw_method=my_kde_bandwidth)

    x_eval = hist[1][0:255]
    pde = kde(x_eval)

    #find the local maxima in the pde using argrelextrema()
    maximum = x_eval[argrelextrema(pde,np.greater)]

    if debug:
        if deep_debug:
            # for formating strings Refer https://pyformat.info/
            # print('{}{:d}{}{}'.format('                U. A. peaks at pixel intensity: ', maximum, ' for ', ROI))
            print(f'                U. A. peaks at pixel intensity: {maximum} for {ROI}')

    #find the relative strength of foreground to background. Evaluate this only if there are two peaks corresponding to foreground and background. Else set to default zero.
    rel_strength = 0
   
    if maximum.size==2:
        strength = kde(maximum)
        #sort the strength in ascending order
        strength.sort()
        rel_strength    =   strength[0]/strength[1]

    elif maximum.size ==1:
        #add duplicate of the peak position so that the maximum is 1x2 array for simpler storing
        maximum = np.append(maximum,maximum[0])

    else:
        #if number of points of maxima is greater than 2, then it implies noisy particles are there in ROI. Need to discard ROI by selecting the peaks with first two highest intensity values. and setting relative intensity to zero
        sorted_index = kde(maximum).argsort()
        first_two_peaks = np.array(maximum[sorted_index[-1]],maximum[sorted_index[-2]])
        maximum = first_two_peaks
        
    #initialize a fig
    fig_mod_analysis = []
    
    if displayImages:
        title = f"{x_width} by {y_width}"
        fig_mod_analysis = plotModAnalysis(img_restored, title, hist, x_eval, kde, pde)

    return maximum, rel_strength, fig_mod_analysis 

