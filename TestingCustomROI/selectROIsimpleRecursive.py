# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:55:29 2020

@author: fubar

ALERT! before running this script set the samplename to Training
"""

import os, os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

import cv2

# import sys

# append the KerrPy folder to sys.path

# KerrPy_folder = os.path.abspath(os.path.join(os.curdir, '..','KerrPy'))

# sys.path.append(KerrPy_folder)

from KerrPy.Image.processOpenCV import saveImage, findEdge, saveRestoredImage

from KerrPy.Image.processROI import restoreColorROI

    
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

def saveIteration():
    global i
    global list_pulse_index
    global list_iter_index
    global list_exp_index
    global parent_dir_abs
    
    # read the coordinates
    coordinates = tuple(np.load('coordinates.npy'))
    print(f"coordinates: {coordinates}")  
    
    # read the image
    img = np.load('img.npy')
    
    
    
    pulse_index = list_pulse_index[i]
    iter_index = list_iter_index[i]
    exp_index = list_exp_index[i]
    
    pulse, img_color, windowROI = findEdge(pulse_index, iter_index, exp_index, img, coordinates=coordinates)

    
    img_restored = restoreColorROI(img_color, img, windowROI)
    
    # save the restored image to file
    saveRestoredImage(pulse_index, iter_index, exp_index, img_restored) 
    
    #save the image to file    
    saveImage(pulse_index, iter_index, exp_index, img_color)
    
    i += 1

def selectROIsimple(img):
    
    # save the image to txt
    
    np.save('img.npy',img)
    # global i
    # i += 1
    # print(f"i {i}"
    
    # x = np.arange(100.) / 99
    # y = np.sin(i*x)
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
    
    cid = fig.canvas.mpl_connect('key_press_event', toggle_selector)
    
    
    
images = ['testSmallDomain.png', 'testBigDomain.png']

list_pulse_index = [0,1]
list_iter_index = [0,1]
list_exp_index = [0,1]
parent_dir_abs = os.curdir

n_images = len(images)
i = 0
def iterateImages():
    global i
    if i < n_images:
        img_file = images[i]
        img = cv2.imread(img_file, 0)
        selectROIsimple(img)
    else:
        return
        

if __name__ == '__main__':
    # img = cv2.imread('wololo.jpg')
    # selectROIsimple(img)
    iterateImages()