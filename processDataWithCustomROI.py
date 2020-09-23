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

from globalVariables import save_exps_file, proc_dir, fits_folder, samplename

from KerrPy.Image.processOpenCV import findEdgeCustomROI, saveImage, saveRestoredImage

from KerrPy.Image.processROI import restoreColorCustomROI

from KerrPy.File.processPulse import savePulse
from KerrPy.File.processSpace import saveSpace

from KerrPy.Fits.processData import processDataWithCustomROI

list_counters = []
list_space_shape = [0,0,0,0]
list_pulse_index = []
list_iter_index = []
list_exp_index = []
list_img_file = []


list_counters = [list_space_shape, list_pulse_index, list_iter_index, list_exp_index, list_img_file]

list_counters = processDataWithCustomROI(list_counters)

list_space_shape = list_counters[0]
list_pulse_index = list_counters[1]
list_iter_index = list_counters[2]
list_exp_index = list_counters[3]
list_img_file = list_counters[4]

space = np.zeros((list_space_shape[0], list_space_shape[1], list_space_shape[2], list_space_shape[3]))

# save the space to .npy

# save the space

parent_dir_abs = os.path.abspath(os.curdir)

proc_dir_abs = os.path.abspath(os.path.join(parent_dir_abs, proc_dir))
if not os.path.isdir(proc_dir_abs): os.mkdir(proc_dir_abs)

fits_dir_abs = os.path.abspath(os.path.join(proc_dir_abs,fits_folder))
if not os.path.isdir(fits_dir_abs): os.mkdir(fits_dir_abs)

fits_root = os.path.abspath(os.path.join(fits_dir_abs,samplename))
if not os.path.isdir(fits_root): os.mkdir(fits_root)

space_filename = save_exps_file

space_filepath = os.path.abspath(os.path.join(fits_root,space_filename))
np.save(space_filepath, space)
  




n_images = len(list_img_file)

i = 0

def iterateImages():
    
    global i

    if i < n_images:
        
        img_file = list_img_file[i]
        
        img = cv2.imread(img_file, 0)
        
        selectROIsimple(img, space)
        
    return

def selectROIsimple(img, space):
    
    # save the image to .npy
    
    np.save('img.npy',img)
    

    fig, axes = plt.subplots(1,2)
    axes[0].imshow(img)
    
    def onselect(eclick, erelease):
        "eclick and erelease are matplotlib events at press and release."
        # print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
        # print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
        # print('used button  : ', eclick.button)
        
        # print(toggle_selector.RS.corners)
        
        #write the coordinates
        
        np.save('coordinates.npy',toggle_selector.RS.corners)
        
        # save the image_crop
        x0 = int(eclick.xdata)
        y0 = int(eclick.ydata)
        x1 = int(erelease.xdata)
        y1 = int(erelease.ydata)
        img_crop = img[y0:y1, x0:x1]
        
        axes[1].imshow(img_crop)
        
        np.save('img_crop.npy',img_crop)
        
    
    toggle_selector.RS = RectangleSelector(axes[0], onselect, drawtype='box')
    plt.show()

    # fig.canvas.mpl_disconnect(cid)
    
    fig.canvas.mpl_connect('key_press_event', toggle_selector)
    
    
def saveIteration():
    global i
    print(f"i is {i}")
    global list_pulse_index
    global list_iter_index
    global list_exp_index
    global parent_dir_abs
    global space_filepath
    
    # read the coordinates
    coordinates = np.load('coordinates.npy')
    print(f"coordinates: {coordinates}")  
    
    # read the image
    img_file = 'img.npy'
    img = np.load(img_file)
    
    #load the space
    space = np.load(space_filepath)
    
    # img_crop = np.load('img_crop.npy')
    
    pulse_index = list_pulse_index[i]
    iter_index = list_iter_index[i]
    exp_index = list_exp_index[i]
    
    pulse, img_color, customROI = findEdgeCustomROI(pulse_index, iter_index, exp_index, img, parent_dir_abs)

    # save the params to space
    
    space[exp_index, iter_index, pulse_index] = pulse
    
    # save the space to space_file again!!!
    
    np.save(space_filepath, space)
    
    # restore the image
    
    img_restored = restoreColorCustomROI(img_color, img, customROI)
        
    #save the image to file    
    saveImage(pulse_index, iter_index, exp_index, img_color, parent_dir_abs)
    
    # save the restored image to file
    saveRestoredImage(pulse_index, iter_index, exp_index, img_restored, parent_dir_abs)
    
    
    # pulse = processOpenCV(pulse_index, iter_index, exp_index, img_file, parent_dir_abs)
    
    # save the params
    savePulse(pulse_index, iter_index, exp_index, pulse, parent_dir_abs)
    
    # iterate the counter after the save
    
    i += 1


### TODO all this saving to and loading from .npy files is due to unable to
# pass as arguments to toggle_selector key press event handler!!
# I need to implement a more cleaner code
def toggle_selector(event):
    # toggle_selector.RS.set_active(False)

    print('Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print('RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)

        saveIteration()
        iterateImages()
        
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print('RectangleSelector activated.')
        toggle_selector.RS.set_active(True)
        pass

if __name__ == '__main__':
    iterateImages()
    # saveSpace(space, parent_dir_abs)