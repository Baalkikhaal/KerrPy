# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 18:07:55 2020

@author: fubar
"""
import matplotlib as mpl
mpl.style.use('myMatplotlibStylesheet.mplstyle')
mpl.rcParams['text.usetex'] =  False
import matplotlib.pyplot as plt

import numpy as np

def displayImage(image, title):
    """
        Display the image with given title
        Set cmap in custom stylesheet is 'inferno'
        Use cmap= 'Greys' for consistency with original image.
        For more colormaps,
        Reference:
        https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html
    """
    
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap='Greys')
    return

def displayNormError(image, norm_err):
    """
        In a multiplot, display the image
        and the normalized error for each point on edge
        with respect to fit ellipse
    """
    fig,(ax1,ax2) = plt.subplots(ncols =2 ,nrows =1, figsize=(8,4))
    ax1.set_title("Normalized error of the fit")
    ax1.plot(norm_err, 'k-')
    ax2.set_title("Modal Analysis")
    ax2.imshow(image, cmap='Greys')

def plotROIanalysis(array_y_ROI, array_rel_strength, label):
    """
        Plot the relative strength with ROI to visualize
        the maximization of the peak intensity resolution.
    """
    fig_ROI_analysis, ax = plt.subplots()
    ax.plot(array_y_ROI,array_rel_strength, label = label)
    
    return fig_ROI_analysis

def plotModAnalysis(img_restored, title, hist, x_eval, kde2, pde):
    """
        0. Display the restored image
        1. Plot the histogram fit with the univariate modal analysis
    """
    # plot all the images
    fig_mod_analysis, [ax1, ax2] = plt.subplots(ncols=2,nrows=1, figsize=(8,4))    

    ax1.imshow(img_restored,cmap='Greys')
    title = title
    ax1.set_title(title)
    ax1.set_xticks([]), ax1.set_yticks([])


    # first plot histogram
#        plt.title('Medianized Histogram')
    ax2.set_xticks([]), ax2.set_yticks([])
    bins = hist[1]
    counts = hist[0]
    norm_weights = counts/np.max(counts)
    ax2.hist(bins, bins, weights = norm_weights)   # 
    
    # then plot smooth pde
    xmin = np.min(x_eval) 
    xmax = np.max(x_eval)
    xrange = np.linspace(xmin, xmax, 100)
    smooth_pde = kde2(xrange)
    ax2.plot(xrange, smooth_pde/np.max(smooth_pde),  '-')

    #finally plot the discrete pde corresponding to bins
    ax2.plot(x_eval, pde/np.max(pde), 'o', color ='#C44E52', label ="M. A.")
    
    ax2.legend()
    
    return fig_mod_analysis

