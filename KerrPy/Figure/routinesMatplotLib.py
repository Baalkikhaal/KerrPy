# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 18:07:55 2020

@author: fubar
"""

import os, os.path

import matplotlib as mpl
mpl.style.use('myMatplotlibStylesheet.mplstyle')
mpl.rcParams['text.usetex'] =  True
mpl.use('pdf', force=True)  # set noninteractive backend 'pdf'
# to check the available non gui backends
#   non_gui_backends = mpl.rcsetup.non_interactive_bk
#   print(non_gui_backends)
#   to check gui backends
#   gui_backends = mpl.rcsetup.interactive_bk
#   Reference : https://stackoverflow.com/questions/3285193/how-to-change-backends-in-matplotlib-python
import matplotlib.pyplot as plt

import numpy as np

from globalVariables import debug

from KerrPy.Fits.processVelocity import fitLinear

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

def plotModAnalysis(img_restored, title, hist, x_eval, kde, pde):
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
    bins = hist[1][0:255]
    counts = hist[0]
    norm_weights = counts/np.max(counts)
    ax2.hist(bins, bins, weights = norm_weights)   # 
    
    # then plot smooth pde
    xmin = np.min(x_eval) 
    xmax = np.max(x_eval)
    xrange = np.linspace(xmin, xmax, 100)
    smooth_pde = kde(xrange)
    ax2.plot(xrange, smooth_pde/np.max(smooth_pde),  '-')

    #finally plot the discrete pde corresponding to bins
    ax2.plot(x_eval, pde/np.max(pde), 'o', color ='#C44E52', label ="M. A.")
    
    ax2.legend()
    
    return fig_mod_analysis

def displayHistogram(images, titles):
    """
        Plot the comparison of images after transformation
        
        since this is subplot of (2,1) we need bigger figsize
        
        change the rcParams using dictionary key : values
        However these are global settings.
        
        #TODO temporary styling using plt.style.context()
        Reference:
        https://matplotlib.org/tutorials/introductory/customizing.html#temporary-styling

        return the plot
    """
    
    mpl.rcParams['figure.figsize'] = 6.0,6.0
    mpl.rcParams['figure.subplot.bottom'] = 0.1
    mpl.rcParams['figure.subplot.left'] = 0.1
    mpl.rcParams['figure.subplot.right'] = 0.9
    mpl.rcParams['figure.subplot.top'] = 0.9
    
    fig, axes = plt.subplots(2,2)
    for i in np.arange(2):
        axes[i,0].imshow(images[i*2],'gray')
        axes[i,0].set_title(titles[i*2])
        axes[i,0].set_xticks([])
        axes[i,0].set_yticks([])
        
        axes[i,1].hist(images[i*2].ravel(),128)
        axes[i,1].set_title(titles[i*2+1])
        axes[i,1].set_xticks([])
        axes[i,1].set_yticks([])
        fig.show()
        
    return fig

def saveFig(fig_file, fig):
    """
        Save the fig to fig_file
    """
    fig.savefig(fig_file)

def displayMorphTrans(images, titles):
    """
        Plot the opening and closing
        leading to speckleless image
    """
    mpl.rcParams['figure.figsize'] = 6.0,6.0
    mpl.rcParams['figure.subplot.bottom'] = 0.1
    mpl.rcParams['figure.subplot.left'] = 0.1
    mpl.rcParams['figure.subplot.right'] = 0.9
    mpl.rcParams['figure.subplot.top'] = 0.9
    
    fig, axes = plt.subplots(2,3)
    
    for i in np.arange(2):
        for j in np.arange(3):
            axes[i,j].imshow(images[i][j],'gray')
            axes[i,j].set_title(titles[i][j])
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
    
    fig.show()

    return fig

def displaySubPlots(images, titles):
    """
        Display the subplots
        
    """
    
    mpl.rcParams['figure.figsize'] = 4.0,2.0
    mpl.rcParams['figure.subplot.bottom'] = 0.1
    mpl.rcParams['figure.subplot.left'] = 0.1
    mpl.rcParams['figure.subplot.right'] = 0.90
    mpl.rcParams['figure.subplot.top'] = 0.90
    
    fig, axes = plt.subplots(1,2)
    
    for i in np.arange(2):
        axes[i].imshow(images[i],'gray')
        axes[i].set_title(titles[i])
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    return fig

def plotIteration(exp_index, iter_index, iteration, control):
    """
        Plot separate plots for confidence, semimajor, semi minor,
        x_center, y_center, orientation for an iteration.
        Iteration consists of pulses. So the variation is 
        across the pulses
        
        Return the figures for the plots.
        
        # TODO plot linear fits to the data
    """
    
    # extracts the controls for the iteration
    Hip, Hop, delta_t = control
    

    
    #########figure plotting##############
    
    # set the label for the figure
    label = f'E: {exp_index} = ' + r'$H_{x}, H_{z}, \Delta t =~$' + f"{Hip}, {Hop}, {delta_t} I: {iter_index}"


    # initialize list of figures.
    list_figs = []
    
    # initialize list of figure filenames
    list_figs_files = []
    
    # labels for each of the figures
    #1D list of strings with rows as channels and columns as title, xlabel, ylabel
    list_labels_1D = [
                        [r'Confidence', r'$\mathrm{n_{pulse}}$', 
                             r'Confidence $c~(\mathrm{in}~\%)$', r'Confidence'],
                        [r'X-Center', r'$\mathrm{n_{pulse}}$',
                             r'$x_{c} $(in pixels)', r'X-Center'],
                        [r'Y-Center', r'$\mathrm{n_{pulse}}$', 
                             r'$y_{c} $(in pixels)', r'Y-Center'], 
                        [r'Semi Major Axis', r'$\mathrm{n_{pulse}}$',
                             r'$a$ (in pixels)', r'Semi-Major-Axis'],
                        [r'Semi Minor Axis', r'$\mathrm{n_{pulse}}$',
                             r'$b$ (in pixels)', r'Semi-Minor-Axis'],
                        [r'Orientation', r'$\mathrm{n_{pulse}}$',
                             r'Orientation $o~(\mathrm{in}~^{\circ})$', r'Orientation']
                    ]

    # loop over the 6 parameters c, a, b, x_c, y_c, o 
    for i in np.arange(6):
        
        #initialize a figure
        fig, ax = plt.subplots(1,1)
        
        mpl.style.use('myMatplotlibStylesheet.mplstyle')
        mpl.rcParams['figure.subplot.right'] = 0.9
        mpl.rcParams['figure.subplot.top'] = 0.8
        mpl.rcParams['legend.fontsize'] = 6
        
        
        ax.set_title(list_labels_1D[i][0])
        ax.set_xlabel(list_labels_1D[i][1])
        ax.set_ylabel(list_labels_1D[i][2])
        
        # set the xticks
        nImages = iteration.shape[0]
        ax.set_xlim(0, nImages)
        xticks = np.arange(0,nImages,1)
        ax.set_xticks(xticks)
        
        x = np.arange(nImages)
        y = iteration[:,i]
        
        ax.plot(x, y, 'o', label = label)
        
        # Place a legend above the subplot,
        
        ax.legend(bbox_to_anchor=(0., 1.12, 1., .102), 
                   loc='lower left', ncol=1, mode="expand", borderaxespad=0.)

        
        #append the figure to list
        list_figs.append(fig)

        #append the fig filename to list
        list_figs_files.append(list_labels_1D[i][3])
        
    ######### Linear fitting ####################
    
    # extract the linear fit parameters (slope, intercept) 
    # to the iteration parameters (x_c, y_c, a, b)
    
    # number of images
    nImages = iteration.shape[0]
    
    x = np.arange(nImages)
    
    y = iteration[:, [1,2,3,4]]
    
    fits = fitLinear(x,y)    
    
    # set the indices of list_figs to draw the lines onto
    
    indices = [1, 2, 3, 4]
    
    # draw line on each of the parameters x_c, y_c, a, b
    
    for i in np.arange(len(indices)):
        
        m = fits[i,0]
        
        c = fits[i,1]
        
        #extract already generated figure object from the list_figs
        
        fig_linear = list_figs[indices[i]]
        
        # axes of a figure are stored as list fig.axes
        ax_linear = fig_linear.axes[0]
        
        x_linear = x
        
        y_linear = m * x + c
        
        ax_linear.plot(x_linear, y_linear, '-')
        
        
    return list_figs, list_figs_files


def plotExperiment(exp_index, avg_iteration, control):
    """
        Plot separate plots for average value of
        confidence, semimajor, semi minor,
        x_center, y_center, orientation over all
        the iterations of an experiment.
        Experiment consists of iterations. So the error
        bars is over the iterations
        
        Return the figures for the plots.
        #TODO place error bars
    """
    
    # extracts the controls for the iteration
    Hip, Hop, delta_t = control
    
    # set the label for the figure
    label = f'E: {exp_index} = '  + r'$H_{x}, H_{z}, \Delta t =~$' + f"{Hip}, {Hop}, {delta_t}"


    # initialize list of figures.
    list_figs = []
    
    # initialize list of figure filenames
    list_figs_files = []
    
    # labels for each of the figures
    #1D list of strings with rows as channels and columns as title, xlabel, ylabel
    list_labels_1D = [
                        [r'Confidence', r'$\mathrm{n_{pulse}}$', 
                             r'Confidence $c~(\mathrm{in}~\%)$', r'Confidence'],
                        [r'X-Center', r'$\mathrm{n_{pulse}}$',
                             r'$x_{c} $(in pixels)', r'X-Center'],
                        [r'Y-Center', r'$\mathrm{n_{pulse}}$', 
                             r'$y_{c} $(in pixels)', r'Y-Center'], 
                        [r'Semi Major Axis', r'$\mathrm{n_{pulse}}$',
                             r'$a$ (in pixels)', r'Semi-Major-Axis'],
                        [r'Semi Minor Axis', r'$\mathrm{n_{pulse}}$',
                             r'$b$ (in pixels)', r'Semi-Minor-Axis'],
                        [r'Orientation', r'$\mathrm{n_{pulse}}$',
                             r'Orientation $o~(\mathrm{in}~^{\circ})$', r'Orientation']
                    ]

    # loop over the 6 parameters c, a, b, x_c, y_c, o 
    for i in np.arange(6):
        
        #initialize a figure
        fig, ax = plt.subplots(1,1)
        
        mpl.style.use('myMatplotlibStylesheet.mplstyle')
        mpl.rcParams['figure.subplot.right'] = 0.9
        mpl.rcParams['figure.subplot.top'] = 0.8
        mpl.rcParams['legend.fontsize'] = 6
        
        
        ax.set_title(list_labels_1D[i][0])
        ax.set_xlabel(list_labels_1D[i][1])
        ax.set_ylabel(list_labels_1D[i][2])
        
        # set the xticks
        nImages = avg_iteration.shape[0]
        ax.set_xlim(0, nImages)
        xticks = np.arange(0,nImages,1)
        ax.set_xticks(xticks)
        
        x = np.arange(nImages)
        y = avg_iteration[:,i]
        
        ax.plot(x, y, 'o', label = label)
        
        # Place a legend above the subplot,
        
        ax.legend(bbox_to_anchor=(0., 1.12, 1., .102), 
                   loc='lower left', ncol=1, mode="expand", borderaxespad=0.)

        
        #append the figure to list
        list_figs.append(fig)

        #append the fig filename to list
        list_figs_files.append(list_labels_1D[i][3])
        
    ######### Linear fitting ####################
    
    # extract the linear fit parameters (slope, intercept) 
    # to the iteration parameters (x_c, y_c, a, b)
    
    # number of images
    nImages = avg_iteration.shape[0]
    
    x = np.arange(nImages)
    
    y = avg_iteration[:, [1,2,3,4]]
    
    fits = fitLinear(x,y)    
    
    # set the indices of list_figs to draw the lines onto
    
    indices = [1, 2, 3, 4]
    
    # draw line on each of the parameters x_c, y_c, a, b
    
    for i in np.arange(len(indices)):
        
        m = fits[i,0]
        
        c = fits[i,1]
        
        #extract already generated figure object from the list_figs
        
        fig_linear = list_figs[indices[i]]
        
        # axes of a figure are stored as list fig.axes
        ax_linear = fig_linear.axes[0]
        
        x_linear = x
        
        y_linear = m * x + c
        
        ax_linear.plot(x_linear, y_linear, '-')


    return list_figs, list_figs_files


def plotSpace(avg_space, controls):
    """
        Plot separate 2D plots for average value of
        confidence, semimajor, semi minor,
        x_center, y_center, orientation over all
        the experiment of a space.
        
        Space consists of experiments. 
        
        Return the figures for the plots.
        #TODO place error bars
    """
    

    # initialize list of figures.
    list_figs = []
    
    # initialize list of figure filenames
    list_figs_files = []
    
    # labels for each of the figures
    #2D list of strings with rows as channels and columns as title, xlabel, ylabel
    labels_2D = [
                    [r'Confidence $c~(\mathrm{in}~\%)$', r'$\mathrm{n_{pulse}}$',
                             r'$[ H_{x}, H_{z}, \Delta t ]$  (in [Oe, Oe, ms])', r'Confidence'],
                    [r'X-Center $x_{c} $(in pixels)', r'$\mathrm{n_{pulse}}$',
                             r'$[ H_{x}, H_{z}, \Delta t ]$  (in [Oe, Oe, ms])', r'X-Center'],
                    [r'Y-Center $y_{c} $(in pixels)', r'$\mathrm{n_{pulse}}$',
                             r'$[ H_{x}, H_{z}, \Delta t ]$  (in [Oe, Oe, ms])',r'Y-Center'],
                    [r'Semi Major Axis $a$ (in pixels)', r'$\mathrm{n_{pulse}}$',
                             r'$[ H_{x}, H_{z}, \Delta t ]$  (in [Oe, Oe, ms])', r'Semi-Major-Axis'],
                    [r'Semi Minor Axis $b$ (in pixels)', r'$\mathrm{n_{pulse}}$',
                             r'$[ H_{x}, H_{z}, \Delta t ]$  (in [Oe, Oe, ms])',r'Semi-Minor-Axis'], 
                    [r'Orientation $o~(\mathrm{in}~^{\circ})$', r'$\mathrm{n_{pulse}}$',
                             r'$[ H_{x}, H_{z}, \Delta t ]$  (in [Oe, Oe, ms])', r'Orientation']
                ]

    if debug: print(f"Plotting 2D plot for the subspace of experiment given by \n {controls}")

    # loop over the 6 parameters c, a, b, x_c, y_c, o 
    for i in np.arange(6):
        
        #surface is made of experiments and images
        # ALERT! So it is necessary avg_space has iteration dimension
        # of size 1
        surface3D = avg_space[:,0,:,:]
        
        # number of experiments
        n_exp = surface3D.shape[0]
        
        #from the above 3D surface, we need extract
        # 2D surface for each parameter c, x_c, y_c, a, b, o
        surface2D = surface3D[:,:,i]
        
        # number of images
        nImages = surface2D.shape[1]
        
        #initialize a figure
        fig, ax = plt.subplots(1,1)
        
        mpl.rcParams['figure.figsize'] = 6.0, 6.0
        mpl.rcParams['figure.subplot.left'] = 0.4
        mpl.rcParams['figure.subplot.right'] = 0.9
        mpl.rcParams['figure.subplot.top'] = 0.8
        
        #disable the minor ticks as they are NOT needed to interpret the surface plot
        mpl.rcParams['xtick.minor.visible'] = False
        mpl.rcParams['ytick.minor.visible'] = False
        
        ax.set_title(labels_2D[i][0]) 
        ax.set_xlabel(labels_2D[i][1])
        ax.set_ylabel(labels_2D[i][2])
        ax.plot(surface2D)
        
        xticks = list(np.arange(nImages))
        yticks = list(controls[:,0])
        ax.set_xticks(np.arange(nImages))
        ax.set_xticklabels(xticks)
        ax.set_yticks(np.arange(n_exp))
        ax.set_yticklabels(yticks)
        
        # create surface plot
        # https://matplotlib.org/gallery/ticks_and_spines/colorbar_tick_labelling_demo.html
        cax = ax.imshow(surface2D)
        
        #set the colorbar
        fig.colorbar(cax)

        
        #append the figure to list
        list_figs.append(fig)
        
        # since we can extract anysubspace let us add
        # prefix reflecting all the experiments in the subspace
        
        prefix = ''
        for j in np.arange(n_exp):
            prefix += f'{controls[j][1]}_'

        suffix = labels_2D[i][3]
        
        fits_2D_filename = f'{prefix}_{suffix}'
        
        #append the fig filename to list
        list_figs_files.append(fits_2D_filename)
    
    print(list_figs_files)

    return list_figs, list_figs_files


