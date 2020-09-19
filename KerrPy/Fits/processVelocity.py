# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 17:45:22 2020

@author: fubar
"""

import os,os.path
import numpy as np

from globalVariables import debug, proc_dir, vel_folder, samplename

from KerrPy.Fits.loadFits import loadFits
from KerrPy.Fits.filterSpace import filterSpace



def saveVelocity( velocity, prefix, parent_dir_abs):
    """
        0. Save velocity array
        1. Save the iteration parameters as csv file.
    """
    
    #Store the current location before relocating
    cur_path = os.path.abspath(os.curdir)
    
    if debug: print("L0 saveVelocity() started")

    proc_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, proc_dir))
    if not os.path.isdir(proc_dir_abs): os.mkdir(proc_dir_abs)
    
    vel_dir_abs = os.path.abspath(os.path.join(proc_dir_abs, vel_folder))
    if not os.path.isdir(vel_dir_abs): os.mkdir(vel_dir_abs)

    vel_root = os.path.abspath(os.path.join(vel_dir_abs,samplename))
    if not os.path.isdir(vel_root): os.mkdir(vel_root)
    
    #change to images root folder (LEVEL 0)
    os.chdir(vel_root)

    # save the velocity array
    vel_file = f"velocity_{prefix}.csv"
    
    header_vel_csv = "Hix (Oe),Hop (Oe),tp (ms),exp_index,V_{xc} (\mu m s^{-1}) , V_{yc} (\mu m s^{-1}), V_{a} (\mu m s^{-1}), b (\mu m s^{-1})"
    
    np.savetxt(vel_file, velocity, delimiter=',', header=header_vel_csv)
    
    #Restore the path to the image path
    os.chdir(cur_path)

def findAverageSpace(space):
    """
        Find the average of parameters for each experiment
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
        
    return avg_space


def findVelocity(avg_space):
    """
        Find the velocity of x_c, y_c, a, b
        for each experiment in the filtered_avg_space
    """

    
    n_exp = avg_space.shape[0]
    
    # initialize an array n_exp x 4 with columns for 
    # velocity of x_c, y_c, a, b 
    
    velocity_space = np.zeros((n_exp, 4))
    
    # number of images
    nImages = avg_space.shape[2]
    
    for i in np.arange(n_exp):
        
        avg_experiment = avg_space[i]
        
        x = np.arange(nImages)
        
        # collect the x_c, y_c, a, b  of experiment
        y = np.transpose(avg_experiment[0,:,[1,2,3,4]])
        
        fits = fitLinear(x, y)
        
        # velocity is proportional to slope of fit
        velocity_space[i] = fits[:,0]
    
    return velocity_space

def fitLinear(x,y):
    """
        0. Use np.polyfit to linear fit the data
        1. Mask the np.nan data using np.isfinit()
        
        Reference:
        https://stackoverflow.com/questions/28647172/numpy-polyfit-doesnt-handle-nan-values
        
        # TODO extract error of fit
    """
    # 4 parameters  x_c, y_c, a, b
    n_params = y.shape[1]
    
    # initialize array for slope, intercept
    fits = np.zeros((n_params, 2))
    
    for i in np.arange(n_params):
        param = y[:,i]
        #indices of x and y which are finite
        idx = np.isfinite(x)
        idy = np.isfinite(param)
        
        # select points with both proper coordinates 
        idxy = idx & idy
        
        # pick the finite coordinates
        x_proper = x[idxy]
        y_proper = param[idxy]
        
        n_proper = len(x_proper)
        
        # initialize slope and intercept
        slope = 0.0
        intercept = 0.0
        # pass the fit if there are less than 3 points
        
        if n_proper < 3:
            if debug : print(f"Insufficient number of points: {n_proper} for linear fit")
            slope = np.nan
            intercept = np.nan
            
        else:
            
            # use np.polyfit return value is p[0] = slope , p[1] = intercept
            p = np.polyfit(x_proper, y_proper, 1)
            slope = p[0]
            intercept = p[1]
            
        fits[i] = [slope, intercept]
        
    if debug: print(f"fits are {fits}")
            
    return fits 
        
 
def scaleVelocity(velocity_space, controls):
    """
        0. Store the experiment index and controls
        
        1.Scale the velocity from pixels per pulse units
            to micrometer per sec
        2. Concatenate the prefix containing experiment indices
            for the filename
    """
    
    
    # initialize array for index and controls
    # velocity in per micrometer per sec 
    # so total 8 columns
    
    velocity_um_per_sec = np.zeros((velocity_space.shape[0], 8))
    
    # initialize prefix for the filename to store the velocity array in
    prefix = ''
    
    n_vel = velocity_um_per_sec.shape[0]
    
    for i in np.arange(n_vel):
        
        
        # controls of the experiment
        
        Hip = np.float(controls[i][0][0])
        Hop = np.float(controls[i][0][1])
        tp = np.float(controls[i][0][2])
        exp_index = controls[i][1]
        
        velocity_um_per_sec[i, 0:4] = [Hip, Hop, tp, exp_index]
        
        # multiply each experiment with corresponding pulse width
        # to convert from per pulse to per ms
        
        velocity_pixel_per_msec = velocity_space[i] / tp
        
        # multiply with scale
        # to convert from pixels to micrometer
        
        scale_space = 100/304 # in um per pixel
        
        velocity_um_per_msec = velocity_pixel_per_msec * scale_space
        
        # multiply with scale
        # to convert from um per msec to um to sec
        
        scale_time = 1000 # in msec per sec
        
        velocity_um_per_sec[i,4:] = velocity_um_per_msec * scale_time
        
        # concatenate the experiment index
        
        prefix += f'{controls[i][1]}_'
        
    
    return velocity_um_per_sec, prefix
           

def processVelocity(parent_dir_abs):
    """
        find the velocity of the ellipse
        
        #TODO add global magnification dictionary
    """
    
    # extract controls and space
    controls, space = loadFits()
    
    # filter the space
    filtered_space = filterSpace(space)
    
    filtered_avg_space = findAverageSpace(filtered_space)
    
    velocity_pixel_per_pulse = findVelocity(filtered_avg_space)
    
    velocity_um_per_sec, vel_file_prefix = scaleVelocity(velocity_pixel_per_pulse, controls)
    
    saveVelocity(velocity_um_per_sec, vel_file_prefix, parent_dir_abs)
    
    return velocity_um_per_sec
    

        

if __name__ == '__main__':
    
    parent_dir_abs = os.path.abspath(os.curdir)

    velocity_um_per_sec = processVelocity(parent_dir_abs)
    
