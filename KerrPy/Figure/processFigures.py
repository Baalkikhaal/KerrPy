# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:12:58 2020

@author: fubar
"""

import os, os.path
import numpy as np


from globalVariables import debug, proc_dir, figs_folder, samplename
from globalVariables import saveFigures, min_confidence

from KerrPy.File.loadFits import loadFits
from KerrPy.Figure.routinesMatplotLib import saveFig, plotIteration, plotExperiment, plotSpace
 



def extractSubspace(list_exp_indices, space ):
    """
        Take inputs as a list of indices for the subspace,
        
        and return the subspace
    """

    subspace = space[list_exp_indices]
    
    return subspace


def filterSpace(space):
    """
        Compare the confidence of fits across the space
        with global min_confidence
        Define badfits as ones with confidence less than
        minimum confidence and extract the indices for
        such pulses.
        
        Set all the pulse parameters corresponding to badfits
        to np.nan
        
        Return the filtered space
    """
    confidence = space[:,:,:,0]
    
    badfits = confidence < min_confidence
    badfits_indices = np.nonzero(badfits)
    
    # ALERT! create a hard copy of space to avoid manipulating it.
    filtered_space = space.copy()
    
    # set the pulse parameters i.e. confidence, a, b, x_c, y_c, o to np.nan
    filtered_space[badfits_indices] = np.nan
    
    return filtered_space

    
def saveFigsSpace( avg_space, list_figs_files, list_figs, parent_dir_abs):
    """
        0. Save list of figs
        1. Save the iteration parameters as csv file.
    """
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    if debug: print("L0 saveFigsSpace() started")

    proc_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, proc_dir))
    if not os.path.isdir(proc_dir_abs): os.mkdir(proc_dir_abs)
    
    figs_dir_abs = os.path.abspath(os.path.join(proc_dir_abs, figs_folder))
    if not os.path.isdir(figs_dir_abs): os.mkdir(figs_dir_abs)

    figs_root = os.path.abspath(os.path.join(figs_dir_abs,samplename))
    if not os.path.isdir(figs_root): os.mkdir(figs_root)
    
    #change to images root folder (LEVEL 0)
    os.chdir(figs_root)

    prefix = "Space_"
    
    for i in np.arange(6):
        
        fig = list_figs[i]
        suffix = list_figs_files[i]
        
        fig_filename = f"{prefix}_{suffix}"
        fig_filename_png = f"{fig_filename}.png"
        fig_filename_pdf = f"{fig_filename}.pdf"

        
        saveFig(fig_filename_png, fig)
        saveFig(fig_filename_pdf, fig)
        

        # save the space params as experiment by images as .csv file
        exp_params_filename = f"{prefix}_{suffix}_parameters.csv"
        surface2D = avg_space[:,0,:,i]
    
        # number of images
        nImages = surface2D.shape[1]
    
        header_iteration_csv = f'{suffix}\n'
    
        for i in np.arange(nImages):
            image_index = i
            header_iteration_csv += f"Pulse {image_index},"

        np.savetxt(exp_params_filename, surface2D, delimiter=',', header=header_iteration_csv)
    
    #Restore the path to the image path
    os.chdir(cur_path)


def saveFigsExperiment(exp_index, avg_iteration, list_figs_files, list_figs, parent_dir_abs):
    """
        0. Save list of figs
        1. Save the iteration parameters as csv file.
    """
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    if debug: print(f"    L1 saveFigsExperiment() started at E:{exp_index}")

    proc_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, proc_dir))
    if not os.path.isdir(proc_dir_abs): os.mkdir(proc_dir_abs)
    
    figs_dir_abs = os.path.abspath(os.path.join(proc_dir_abs, figs_folder))
    if not os.path.isdir(figs_dir_abs): os.mkdir(figs_dir_abs)

    figs_root = os.path.abspath(os.path.join(figs_dir_abs,samplename))
    if not os.path.isdir(figs_root): os.mkdir(figs_root)
    
    #change to images root folder (LEVEL 0)
    os.chdir(figs_root)

    # folder for current experiment; create it if not yet
    fig_exp_folder = f"Experiment_{exp_index}"
    if not os.path.isdir(fig_exp_folder): os.mkdir(fig_exp_folder)
    
    #change to images experiment folder (LEVEL 1)
    os.chdir(fig_exp_folder)

    # create header for iteration parameters .csv file
    prefix = f"Experiment_{exp_index}"
    header_iteration_csv = ''
    
    for i in np.arange(6):
        
        fig = list_figs[i]
        suffix = list_figs_files[i]
        
        fig_filename = f"{prefix}_{suffix}"
        fig_filename_png = f"{fig_filename}.png"
        fig_filename_pdf = f"{fig_filename}.pdf"

        header_iteration_csv += f"{suffix},"
        
        saveFig(fig_filename_png, fig)
        saveFig(fig_filename_pdf, fig)
        
    # save the iteration params as .csv file
    exp_params_filename = f"{prefix}_parameters.csv"
    np.savetxt(exp_params_filename, avg_iteration, delimiter=',', header=header_iteration_csv)
    
    #Restore the path to the image path
    os.chdir(cur_path)


def saveFigsIteration( iter_index, exp_index, iteration, list_figs_files, list_figs, parent_dir_abs):
    """
        0. Save list of figs
        1. Save the iteration parameters as csv file.
    """
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    if debug: print(f"        L2 saveFigList() started at E:{exp_index} I:{iter_index}")

    proc_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, proc_dir))
    if not os.path.isdir(proc_dir_abs): os.mkdir(proc_dir_abs)
    
    figs_dir_abs = os.path.abspath(os.path.join(proc_dir_abs, figs_folder))
    if not os.path.isdir(figs_dir_abs): os.mkdir(figs_dir_abs)

    figs_root = os.path.abspath(os.path.join(figs_dir_abs,samplename))
    if not os.path.isdir(figs_root): os.mkdir(figs_root)
    
    #change to images root folder (LEVEL 0)
    os.chdir(figs_root)

    # folder for current experiment; create it if not yet
    fig_exp_folder = f"Experiment_{exp_index}"
    if not os.path.isdir(fig_exp_folder): os.mkdir(fig_exp_folder)
    
    #change to images experiment folder (LEVEL 1)
    os.chdir(fig_exp_folder)

    # folder for current iteration; create it if not yet
    fig_iter_folder = f"Experiment_{exp_index}_Iteration_{iter_index}"
    if not os.path.isdir(fig_iter_folder): os.mkdir(fig_iter_folder)

    #change to images iteration folder (LEVEL 2)
    os.chdir(fig_iter_folder)

    # create header for iteration parameters .csv file
    prefix = f"Experiment_{exp_index}_Iteration_{iter_index}"
    header_iteration_csv = ''
    
    for i in np.arange(6):
        
        fig = list_figs[i]
        suffix = list_figs_files[i]
        
        fig_filename = f"{prefix}_{suffix}"
        fig_filename_png = f"{fig_filename}.png"
        fig_filename_pdf = f"{fig_filename}.pdf"

        header_iteration_csv += f"{suffix},"
        
        saveFig(fig_filename_png, fig)
        saveFig(fig_filename_pdf, fig)
        
    # save the iteration params as .csv file
    iter_params_filename = f"{prefix}_parameters.csv"
    np.savetxt(iter_params_filename, iteration, delimiter=',', header=header_iteration_csv)
    
    #Restore the path to the image path
    os.chdir(cur_path)
    

def drawIterationFigures(space, controls, parent_dir_abs):
    """
        0. Iterate over all the iterations of all experiments
        
        1. Save the figures to global figs_folder
    
        ALERT! is called only when the saveFigures is True
    """

    
    #number of experiments
    n_exp = space.shape[0]
    
    for i in np.arange(n_exp):
        
        exp_index = i
        experiment = space[i]
        control = controls[i][0]
        
        # number of iterations
        n_iter = experiment.shape[0]
        
        for j in np.arange(n_iter):
            
            iter_index = j
            iteration = experiment[j]
            
            #plot the figures of the iteration
            list_figs, list_figs_files = plotIteration(exp_index, iter_index, iteration, control)
            
            # save the list of figures and iteration parameters
            
            saveFigsIteration(iter_index, exp_index, iteration, list_figs_files, list_figs, parent_dir_abs)
            
            
        

        
def drawExperimentFigures(space, controls, parent_dir_abs):
    """
        Plot the average values for the parameters
        over the iterations
        
        # TODO insert error bars
    """
    n_exp = space.shape[0]
    
    for i in np.arange(n_exp):
        exp_index = i
        experiment = space[i]
        control = controls[i][0]
        
        # number of iterations
        n_iter = experiment.shape[0]
        
        # initialize average iteration
        avg_iteration = np.zeros(experiment[0].shape, dtype = np.float)
        
        for j in np.arange(n_iter):
            
            avg_iteration += experiment[j]
        
        # average iteration
        avg_iteration /= n_iter
        
        #plot the figures of the experiment
        list_figs, list_figs_files = plotExperiment(exp_index, avg_iteration, control)

        
        # save the list of figures and iteration parameters
            
        saveFigsExperiment(exp_index, avg_iteration, list_figs_files, list_figs, parent_dir_abs)
        
    

def drawSpaceFigures(space, controls, parent_dir_abs):
    """
        Plot 2D surface plots for each parameter over the entire
        space
        
        # TODO insert error bars
    """
    n_exp = space.shape[0]
    
    # initialize average space.
    # average space is reduced space with iterations collapsed 
    # to single average iteration
    
    avg_space = np.zeros((n_exp, 1, space.shape[2], space.shape[3]))
    
    for i in np.arange(n_exp):
        experiment = space[i]
        
        # number of iterations
        n_iter = experiment.shape[0]
        
        # initialize average iteration
        avg_iteration = np.zeros(experiment[0].shape, dtype = np.float)
        
        for j in np.arange(n_iter):
            
            avg_iteration += experiment[j]
        
        # average iteration
        avg_iteration /= n_iter
        
        # set the average iteration to average space
        
        avg_space[i, 0] = avg_iteration
        
        #plot the figures of the experiment
        list_figs, list_figs_files = plotSpace(avg_space, controls)
    
        # save the list of figures and experiment parameters
            
        saveFigsSpace( avg_space, list_figs_files, list_figs, parent_dir_abs)


def processFigures(parent_dir_abs):
    """
        Plot the figures and save the figures
    """
    # extract controls and space
    controls, space = loadFits()
    
    # filter the space
    filtered_space = filterSpace(space)
    
    # plot the figures
    if saveFigures:
        drawIterationFigures(filtered_space, controls, parent_dir_abs)
        drawExperimentFigures(filtered_space, controls, parent_dir_abs)
        drawSpaceFigures(filtered_space, controls, parent_dir_abs)
        
    return controls, filtered_space
    
    
if __name__ == "__main__":
    
    parent_dir_abs = os.path.abspath(os.curdir)
    
    processFigures(parent_dir_abs)
    
