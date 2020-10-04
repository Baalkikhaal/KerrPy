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
from globalVariables import mpl_stylesheet, use_tex

mpl.style.use(mpl_stylesheet)
mpl.rcParams['text.usetex'] =  use_tex

import matplotlib.pyplot as plt

from matplotlib.widgets import RectangleSelector

import cv2

from globalVariables import deep_debug

from KerrPy.Image.processOpenCV import findEdgeCustomROI, saveImage, saveRestoredImage

from KerrPy.Image.processROI import restoreColorCustomROI

from KerrPy.File.processPulse import savePulse

from KerrPy.Fits.processData import processDataWithCustomROI

from KerrPy.File.processSpace import saveSpace


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
space = np.zeros((list_space_shape[0], list_space_shape[1], list_space_shape[2], list_space_shape[3]), dtype=np.float)

# save the space
parent_dir_abs = os.path.abspath(os.curdir)

# number of images
n_images = len(list_img_file)


def iterateImages():
    
    # get the counter of the function
    i = iterateImages.counter

    if i < n_images:
        
        img_file = list_img_file[i]
            
        img = cv2.imread(img_file, 0)
        
        # set the img attribute of iterateImages
        iterateImages.img = img
        
        myWidget.ax.figure.axes[0].imshow(img)
        
        #get attribute axes
        
        #update the axes using update method
        myWidget.update()

    else:
        #deactivate the window
        myWidget.set_active(False)
        
        # save the space
        space = iterateImages.space
        
        saveSpace(space, parent_dir_abs)
        
    
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
    
    coordinates = myWidget.corners
    
    # update the attribute of iterateImage coordinate
    iterateImages.coordinates = coordinates
    
    # get the image attribute
    img = iterateImages.img
    
    # save the image_crop
    x0 = int(eclick.xdata)
    y0 = int(eclick.ydata)
    x1 = int(erelease.xdata)
    y1 = int(erelease.ydata)
    img_crop = img[y0:y1, x0:x1]
    
    # write the img crop to cropped axis 
    myWidget.ax.figure.axes[1].imshow(img_crop)
    
    # update the axes
    myWidget.update()
    
    #update the restored image
    updateRestoredImage(img, coordinates)

    return

def updateRestoredImage(img, coordinates):
    # read the coordinates
    # coordinates = np.load(coordinates_file)
    if deep_debug: print(f"coordinates: {coordinates}")  
    
    # current counter
    i = iterateImages.counter
    
    pulse_index = list_pulse_index[i]
    iter_index = list_iter_index[i]
    exp_index = list_exp_index[i]
    
    pulse, img_color, customROI = findEdgeCustomROI(coordinates, pulse_index, iter_index, exp_index, img, parent_dir_abs)

    img_restored = restoreColorCustomROI(img_color, img, customROI)
    
    # update the subplot restored image
    myWidget.ax.figure.axes[2].imshow(img_restored)
    myWidget.update()
    
    # update the space attribute
    
    space  = iterateImages.space
    space[exp_index, iter_index, pulse_index] = pulse
    iterateImages.space = space
    
    # update the img_color attribute
    
    iterateImages.img_color = img_color    
    
    # update the img_restored attribute

    iterateImages.img_restored = img_restored
    
    return


def saveIteration():
    global list_pulse_index
    global list_iter_index
    global list_exp_index
    global parent_dir_abs
    global space_file
    
    
    # get the counter
    i = iterateImages.counter
    print(f"i is {i}")
    

    
    # read the img_color attribute
    img_color = iterateImages.img_color
    
    # get the restored_image attribute
    img_restored = iterateImages.img_restored
    
    #read the space attribute
    space = iterateImages.space
    
    # indices of the image
    pulse_index = list_pulse_index[i]
    iter_index = list_iter_index[i]
    exp_index = list_exp_index[i]
    
    # get the pulse params from space
    pulse = space[exp_index, iter_index, pulse_index] 
    
    
    #save the image to file    
    saveImage(pulse_index, iter_index, exp_index, img_color, parent_dir_abs)
    
    # save the restored image to file
    saveRestoredImage(pulse_index, iter_index, exp_index, img_restored, parent_dir_abs) 
    
    # save the params
    savePulse(pulse_index, iter_index, exp_index, pulse, parent_dir_abs)
    
    # iterate the counter after the save
    iterateImages.counter += 1

    return

def discardIteration():
    global list_pulse_index
    global list_iter_index
    global list_exp_index
    global parent_dir_abs
    global space_file
    
    # get the counter
    i = iterateImages.counter
    print(f"i is {i}")
    
    # read the image attribute
    img = iterateImages.img
    
    #read  the space attribute
    space = iterateImages.space
    
    # indices of the image
    pulse_index = list_pulse_index[i]
    iter_index = list_iter_index[i]
    exp_index = list_exp_index[i]
    
    # set the pulse params to np.nan except confidence which is set to 0
    pulse = np.array([0, np.nan, np.nan, np.nan, np.nan, np.nan])
    
    # update the pulse params to space
    space[exp_index, iter_index, pulse_index] = pulse
    iterateImages.space = space
    
    #save the image to file    
    saveImage(pulse_index, iter_index, exp_index, img, parent_dir_abs)
    
    # save the restored image to file
    saveRestoredImage(pulse_index, iter_index, exp_index, img, parent_dir_abs)
    
    # save the params
    savePulse(pulse_index, iter_index, exp_index, pulse, parent_dir_abs)
    
    # iterate the counter after the save
    iterateImages.counter += 1

    return

def toggle_selector(event):
    """
        TODO all this saving to and loading from .npy files is due to unable to
        pass as arguments to toggle_selector key press event handler!!
        I need to implement a more cleaner code!!
    """
    
    print('Key pressed.')
    if event.key in ['N', 'n'] and myWidget.get_active():
        print('Image iterated.')
        myWidget.set_active(True)

        saveIteration()
        iterateImages()

    if event.key in ['D', 'd'] and myWidget.get_active():
        print('Image discarded.')
        myWidget.set_active(True)

        discardIteration()
        iterateImages()
        
    if event.key in ['A', 'a'] and not myWidget.get_active():
        print('RectangleSelector activated.')
        myWidget.set_active(True)

if __name__ == '__main__':

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



    # get the counter of the function
    i = 0
    
    img_file = list_img_file[i]
        
    img = cv2.imread(img_file, 0)
    
    # show the raw image
    axes[0].imshow(img)
    
    # a reference to the RectangleSelector widget is needed
    # to prevent from Garbage Collection
    myWidget = RectangleSelector(axes[0], onselect)

    # initialize attributes of iterateImages
    iterateImages.counter = 0
    iterateImages.space = space
    iterateImages.img = img
    iterateImages.img_color = img
    iterateImages.img_restored = img 



    # create event loop for matplotlib fig window    
    myWidget.connect_event('key_press_event', toggle_selector)


    
