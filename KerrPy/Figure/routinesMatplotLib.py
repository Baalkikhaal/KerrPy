# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 18:07:55 2020

@author: fubar
"""

import os, os.path

import matplotlib as mpl
mpl.style.use('myMatplotlibStylesheet.mplstyle')
mpl.rcParams['text.usetex'] =  True
import matplotlib.pyplot as plt

import numpy as np

from globalVariables import debug

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
    """
    
    # extracts the controls for the iteration
    Hip, Hop, delta_t = control
    
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
        
        ax.plot(x, y, label = label)
        
        # Place a legend above the subplot,
        
        ax.legend(bbox_to_anchor=(0., 1.12, 1., .102), 
                   loc='lower left', ncol=1, mode="expand", borderaxespad=0.)

        
        #append the figure to list
        list_figs.append(fig)

        #append the fig filename to list
        list_figs_files.append(list_labels_1D[i][3])

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
        
        ax.plot(x, y, label = label)
        
        # Place a legend above the subplot,
        
        ax.legend(bbox_to_anchor=(0., 1.12, 1., .102), 
                   loc='lower left', ncol=1, mode="expand", borderaxespad=0.)

        
        #append the figure to list
        list_figs.append(fig)

        #append the fig filename to list
        list_figs_files.append(list_labels_1D[i][3])

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


def PlotTimeSequence(subspace_experiments, subspace_controls_extracted, controls_extracted, nImages, saveImages, fits_dir_par, samplename, debug):
    n_exp = subspace_experiments.shape[1]# no of experiments in the subspace
    print(f"n_exp is {n_exp}")

    if n_exp == 0:
        if debug: print('Alert!!! No experiment found... Bypassing plotting!')
        pass

    else:
        #2D list of strings with rows as channels and columns as title, xlabel, ylabel
        labels_2D = [[r'Confidence $c~(\mathrm{in}~\%)$', r'$\mathrm{n_{pulse}}$', r'$[ H_{x}, H_{z}, \Delta t ]$  (in [Oe, Oe, ms])', r'Confidence'],[r'X-Center $x_{c} $(in pixels)', r'$\mathrm{n_{pulse}}$', r'$[ H_{x}, H_{z}, \Delta t ]$  (in [Oe, Oe, ms])', r'X-Center'], [r'Y-Center $y_{c} $(in pixels)', r'$\mathrm{n_{pulse}}$', r'$[ H_{x}, H_{z}, \Delta t ]$  (in [Oe, Oe, ms])',r'Y-Center'], [r'Semi Major Axis $a$ (in pixels)', r'$\mathrm{n_{pulse}}$', r'$[ H_{x}, H_{z}, \Delta t ]$  (in [Oe, Oe, ms])', r'Semi-Major-Axis'],[r'Semi Minor Axis $b$ (in pixels)', r'$\mathrm{n_{pulse}}$', r'$[ H_{x}, H_{z}, \Delta t ]$  (in [Oe, Oe, ms])',r'Semi-Minor-Axis'], [r'Orientation $o~(\mathrm{in}~^{\circ})$', r'$\mathrm{n_{pulse}}$', r'$[ H_{x}, H_{z}, \Delta t ]$  (in [Oe, Oe, ms])', r'Orientation']]

        if debug: print("Plotting 2D plot for the subspace of experiment given by \n" + str(subspace_experiments))
        
        for i in np.arange(6):
            mpl.style.use('myMatplotlibStylesheet.mplstyle')
            mpl.rcParams['figure.figsize'] = 3.0, 3.0
            mpl.rcParams['figure.subplot.left'] = 0.4
            mpl.rcParams['figure.subplot.right'] = 0.9
            mpl.rcParams['figure.subplot.top'] = 0.8
            #disable the minor ticks as they are NOT needed to interpret the surface plot
            mpl.rcParams['xtick.minor.visible'] = False
            mpl.rcParams['ytick.minor.visible'] = False
            plt.figure()
            plt.title(labels_2D[i][0]) 
            plt.xlabel(labels_2D[i][1])
            plt.ylabel(labels_2D[i][2])
            plt.plot(subspace_experiments[i])
            xticks = list(np.arange(nImages))
            yticks = list(subspace_controls_extracted)
            plt.xticks(np.arange(nImages), xticks)
            plt.yticks(np.arange(n_exp), yticks)
            plt.imshow(subspace_experiments[i])
            plt.colorbar()
            
            #Store the location of the current image which is under processing
            cur_path = os.path.abspath(os.curdir)
            
            if saveImages:
                # save the experiments, controls_extracted and control knobs 
                if debug: print("Saving 2D plots for " + str(subspace_controls_extracted) + ' ' + str(labels_2D[i][0]))
    
                fits_root = os.path.abspath(os.path.join(fits_dir_par,samplename))
                os.chdir(fits_root)
    
                # for the filename, let us make string of indices of experiments
                subspace_experiments_indices = ExtractSubspaceIndices(subspace_controls_extracted, controls_extracted, debug)
                
                fits_2D_filename = ''
                for each in subspace_experiments_indices: fits_2D_filename += f'{each}_'
                fits_2D_filename_png = f'{fits_2D_filename}_{labels_2D[i][3]}.png'
                fits_2D_filename_pdf = f'{fits_2D_filename}_{labels_2D[i][3]}.pdf'

                # save the plot to the file
                plt.savefig(fits_2D_filename_png)
                plt.savefig(fits_2D_filename_pdf)

                #one times we used os.chdir() so we need to come out of the tree three times
                os.chdir('..')
            
            #Restore the path to the image path
            os.chdir(cur_path)
    return

def ExtractSubspaceIndices(subspace_controls_extracted, controls_extracted, debug):
    if debug: print ("Searching for subspace indices" + str(subspace_controls_extracted))
    subspace_experiments_indices = []
    n = subspace_controls_extracted.shape[0]
    if debug: print("No of subspace experiments found: " + str(n))
    N = controls_extracted.shape[0]
    if debug: print("No of total experiments found: " + str(n))

    for i in np.arange(n):
        for j in np.arange(N):
            if (subspace_controls_extracted[i] == controls_extracted[j]).all():
              if debug:print('Found experiment' + str(controls_extracted[j]))
              subspace_experiments_indices.append(j)
    n_ind = len(subspace_experiments_indices)

    if n_ind == 0:
        if debug: print("Experiment not found!!!")
    #set up the subspace of experiments
    if debug: print(f'Subspace experiment indices are: {subspace_experiments_indices}')
    subspace_experiments_indices = np.asarray(subspace_experiments_indices)
    return subspace_experiments_indices

# def PlotSubspace(filtered_experiments, Hip, Hop):
#     subspace_experiments, subspace_experiments_indices = extractTimeSequenceExperiments(filtered_experiments, controls_extracted, Hip, Hop,nImages, DEBUG=debug)
#     subspace_controls_extracted = controls_extracted[[subspace_experiments_indices]]
#     print(f"subspace controls extracted is {subspace_controls_extracted}")


#     PlotTimeSequence(subspace_experiments, subspace_controls_extracted, controls_extracted, nImages, saveImages, fits_dir_par, samplename, debug)
#     subspace_experimentes_indices = ExtractSubspaceIndices(subspace_controls_extracted, controls_extracted, debug)
# def PlotFullSpace(experiments, controls_extracted):
#     fits_params = experiments.shape[0]
#     n_exp = experiments.shape[1]
#     n_pulse = experiments.shape[2]
        
#     PlotTimeSequence(experiments, controls_extracted, controls_extracted, nImages, saveImages, fits_dir_par, samplename, debug)

# def PlotExperimentIndices(experiments, experiment_indices, controls_extracted):
#     subspace_experiments = experiments[:,experiment_indices,:]
#     subspace_controls_extracted = controls_extracted[experiment_indices]

