# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:19:27 2020

@author: fubar
"""

import os, os.path
import numpy as np

import matplotlib as mpl
#use interactive backend Qt5Agg
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

import cv2

from globalVariables import deep_debug, save_exps_file, proc_dir, fits_folder, samplename

from KerrPy.Image.processOpenCV import findEdgeCustomROI, saveImage, saveRestoredImage

from KerrPy.Image.processROI import restoreColorCustomROI

from KerrPy.File.processPulse import savePulse

from KerrPy.Fits.processData import processDataWithCustomROI


# initialize list for indices of the image
list_counters = []
list_space_shape = [0,0,0,0]
list_pulse_index = []
list_iter_index = []
list_exp_index = []
list_img_file = []
list_counters = [list_space_shape, list_pulse_index, list_iter_index, list_exp_index, list_img_file]


# find the indices 
list_counters = processDataWithCustomROI(list_counters)
list_space_shape = list_counters[0]
list_pulse_index = list_counters[1]
list_iter_index = list_counters[2]
list_exp_index = list_counters[3]
list_img_file = list_counters[4]


# set the shape of space array
space = np.zeros((list_space_shape[0], list_space_shape[1], list_space_shape[2], list_space_shape[3]))

# save the space
parent_dir_abs = os.path.abspath(os.curdir)

proc_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, proc_dir))
if not os.path.isdir(proc_dir_abs): os.mkdir(proc_dir_abs)

fits_dir_abs = os.path.abspath(os.path.join(proc_dir_abs,fits_folder))
if not os.path.isdir(fits_dir_abs): os.mkdir(fits_dir_abs)

fits_root = os.path.abspath(os.path.join(fits_dir_abs,samplename))
if not os.path.isdir(fits_root): os.mkdir(fits_root)

space_file = os.path.abspath(os.path.join(fits_root, save_exps_file))
np.save(space_file, space)
  


###################################################

# create Temp folder at parent dir
temp_dir = os.path.abspath(os.path.join(parent_dir_abs, 'Temp'))
if not os.path.isdir(temp_dir): os.mkdir(temp_dir)

pulse_file = os.path.abspath(os.path.join(parent_dir_abs, temp_dir, 'pulse.npy'))

img_file = os.path.abspath(os.path.join(parent_dir_abs, temp_dir, 'img.npy'))

coordinates_file = os.path.abspath(os.path.join(parent_dir_abs, temp_dir, 'coordinates.npy'))

img_color_file = os.path.abspath(os.path.join(parent_dir_abs, temp_dir, 'img_color.npy'))

img_restored_file = os.path.abspath(os.path.join(parent_dir_abs, temp_dir, 'img_restored.npy'))





# number of images
n_images = len(list_img_file)

#initialize counter
i = 0

def iterateImages():
    
    global i

    if i < n_images:
        
        img_file = list_img_file[i]
        
        img = cv2.imread(img_file, 0)
        
        selectROIrectangle(img)
        
    return

def selectROIrectangle(img):
    """
        Use RectangleSelector class to create ROIrectangle
        Reference : https://matplotlib.org/3.3.1/gallery/widgets/rectangle_selector.html

    """
    
    
    # save the image
    np.save(img_file,img)
    
    # set the matplotlib figure params
    mpl.rcParams['figure.figsize'] = 9.0,4.0
    mpl.rcParams['figure.subplot.bottom'] = 0.1
    mpl.rcParams['figure.subplot.left'] = 0.1
    mpl.rcParams['figure.subplot.right'] = 0.9
    mpl.rcParams['figure.subplot.top'] = 0.7

    fig, axes = plt.subplots(1,3)
    
    
    fig.suptitle('custom ROI selection', fontsize= 24)
    fig.text(0.5,0.85, '''Press `N' to iterate image, `D' to discard image, `Q' to kill the window''',
             ha='center', va='center')
    axes[0].set_title('raw image')
    axes[1].set_title('cropped image')
    axes[2].set_title('restored image')
    
    # show the raw image
    axes[0].imshow(img)
    
    # create event loop for matplotlib fig window    
    fig.canvas.mpl_connect('key_press_event', toggle_selector)

    
    # event handling functions
    def onselect(eclick, erelease):
        """
            This function is called when keypress and release event is created.
            
            eclick and erelease are matplotlib events at press and release.
            
            # print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
            # print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
            # print('used button  : ', eclick.button)
            
            
            However we will use toggle_selector.RS.corners attribute
            to retrieve the four corners of the rectangle
        """
        
        coordinates = toggle_selector.RS.corners
        
        #write the coordinates
        
        np.save(coordinates_file, coordinates)
        
        # save the image_crop
        x0 = int(eclick.xdata)
        y0 = int(eclick.ydata)
        x1 = int(erelease.xdata)
        y1 = int(erelease.ydata)
        img_crop = img[y0:y1, x0:x1]
        
        axes[1].imshow(img_crop)
        
        #update the restored image
        updateRestoredImage()
    
        return

    def updateRestoredImage():
        # read the coordinates
        coordinates = np.load(coordinates_file)
        if deep_debug: print(f"coordinates: {coordinates}")  
        
        # read the image
        img = np.load(img_file)
        
        pulse_index = list_pulse_index[i]
        iter_index = list_iter_index[i]
        exp_index = list_exp_index[i]
        
        pulse, img_color, customROI = findEdgeCustomROI(coordinates, pulse_index, iter_index, exp_index, img, parent_dir_abs)
    
        img_restored = restoreColorCustomROI(img_color, img, customROI)
        
        # update the subplot restored image
        
        axes[2].imshow(img_restored)
        
        # save the pulse
        
        np.save(pulse_file, pulse)
        
        # save the img_color
        
        np.save(img_color_file, img_color)
        
        # save the img_restored
    
        np.save(img_restored_file, img_restored)
        
        return

    
    # toggle selector
    toggle_selector.RS = RectangleSelector(axes[0], onselect, drawtype='box')
    
    fig.show()
    
    return   
    


def saveIteration():
    global i
    print(f"i is {i}")
    global list_pulse_index
    global list_iter_index
    global list_exp_index
    global parent_dir_abs
    global space_file
    
    # read the coordinates
    coordinates = np.load('coordinates.npy')
    print(f"coordinates: {coordinates}")  
    
    # read the pulse
    pulse = np.load(pulse_file)
    
    # read the img_color
    img_color = np.load(img_color_file)
    
    # read the restored_image
    img_restored = np.load(img_restored_file)
    
    #read the space
    space = np.load(space_file)
    
    # indices of the image
    pulse_index = list_pulse_index[i]
    iter_index = list_iter_index[i]
    exp_index = list_exp_index[i]
    
    # save the params to space
    space[exp_index, iter_index, pulse_index] = pulse
    
    # save the space to space_file again!!!
    np.save(space_file, space)
    
    #save the image to file    
    saveImage(pulse_index, iter_index, exp_index, img_color, parent_dir_abs)
    
    # save the restored image to file
    saveRestoredImage(pulse_index, iter_index, exp_index, img_restored, parent_dir_abs) 
    
    # save the params
    savePulse(pulse_index, iter_index, exp_index, pulse, parent_dir_abs)
    
    # iterate the counter after the save
    i += 1

    return

def discardIteration():
    global i
    print(f"i is {i}")
    global list_pulse_index
    global list_iter_index
    global list_exp_index
    global parent_dir_abs
    global space_file
    
    
    # read the image
    img = np.load(img_file)
    
    #load the space
    space = np.load(space_file)
    
    # indices of the image
    pulse_index = list_pulse_index[i]
    iter_index = list_iter_index[i]
    exp_index = list_exp_index[i]
    
    # set the pulse params to np.nan except confidence which is set to 0
    pulse = np.array([0, np.nan, np.nan, np.nan, np.nan, np.nan])
    
    # save the params to space
    space[exp_index, iter_index, pulse_index] = pulse
    
    # save the space to space_file again!!!
    np.save(space_file, space)
    
    #save the image to file    
    saveImage(pulse_index, iter_index, exp_index, img, parent_dir_abs)
    
    # save the restored image to file
    saveRestoredImage(pulse_index, iter_index, exp_index, img, parent_dir_abs)
    
    # save the params
    savePulse(pulse_index, iter_index, exp_index, pulse, parent_dir_abs)
    
    # iterate the counter after the save
    i += 1

    return

def toggle_selector(event):
    """
        TODO all this saving to and loading from .npy files is due to unable to
        pass as arguments to toggle_selector key press event handler!!
        I need to implement a more cleaner code!!
    """
    
    print('Key pressed.')
    if event.key in ['N', 'n'] and toggle_selector.RS.active:
        print('RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)

        saveIteration()
        iterateImages()

    if event.key in ['D', 'd'] and toggle_selector.RS.active:
        print('RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)

        discardIteration()
        iterateImages()
        
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print('RectangleSelector activated.')
        toggle_selector.RS.set_active(True)
        pass

if __name__ == '__main__':
    iterateImages()
