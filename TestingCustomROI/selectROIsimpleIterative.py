# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:55:29 2020

@author: fubar
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from time import sleep

import cv2


    
def toggle_selector(event):
    # toggle_selector.RS.set_active(False)

    print('Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print('RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)

        # read the coordinates
        coordinates = np.load('coordinates.npy')
        print(f"coordinates: {coordinates}")  
        
        img_crop = np.load('img_crop.npy')
        
        
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print('RectangleSelector activated.')
        toggle_selector.RS.set_active(True)
        pass

    

def selectROIsimple(img):
    
    # save the image to txt
    
    np.save('image.npy',img)
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
        
        # reload the image_crop
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
    
    
images = ['wololo.jpg', 'testBigDomain.png']
n_images = len(images)

def iterateImages():
    for i in np.arange(n_images):
        img_file = images[i]
        img = cv2.imread(img_file)
        selectROIsimple(img)
        
iterateImages()
