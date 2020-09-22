# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:55:29 2020

@author: fubar
"""

import os, os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

import cv2

from KerrPy.Image.processOpenCV import findEdgeCustomROI, saveImage, saveRestoredImage, restoreColorCustomROI

from KerrPy.Image.processROI import restoreColorROI

from KerrPy.File.processPulse import savePulse

    
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
    print(f"i is {i}")
    global list_pulse_index
    global list_iter_index
    global list_exp_index
    global parent_dir_abs
    
    # read the coordinates
    coordinates = np.load('coordinates.npy')
    print(f"coordinates: {coordinates}")  
    
    # read the image
    img_file = 'img.npy'
    img = np.load(img_file)
    
    # img_crop = np.load('img_crop.npy')
    
    pulse_index = list_pulse_index[i]
    iter_index = list_iter_index[i]
    exp_index = list_exp_index[i]
    
    pulse, img_color, customROI = findEdgeCustomROI(pulse_index, iter_index, exp_index, img, parent_dir_abs)

    
    img_restored = restoreColorCustomROI(img_color, img, customROI)
        
    #save the image to file    
    saveImage(pulse_index, iter_index, exp_index, img_color, parent_dir_abs)
    
    # save the restored image to file
    saveRestoredImage(pulse_index, iter_index, exp_index, img_restored, parent_dir_abs)
    
    
    # pulse = processOpenCV(pulse_index, iter_index, exp_index, img_file, parent_dir_abs)
    
    # save the params
    savePulse(pulse_index, iter_index, exp_index, pulse, parent_dir_abs)


def selectROIsimple(img):
    
    # save the image to txt
    
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
    
    cid = fig.canvas.mpl_connect('key_press_event', toggle_selector)
    
    
    
images = ['testSmallDomain.png', 'testMediumDomain.png','testBigDomain.png']

list_pulse_index = [0,1,2]
list_iter_index = [0,0,0]
list_exp_index = [0,0,0]
parent_dir_abs = os.curdir

n_images = len(images)
i = -1
def iterateImages():
    
    global i
    i += 1

    if i >= n_images:
        return
    else:
        
        img_file = images[i]
        img = cv2.imread(img_file, 0)
        selectROIsimple(img)
        
    

if __name__ == '__main__':
    # img = cv2.imread('wololo.jpg')
    # selectROIsimple(img)
    iterateImages()