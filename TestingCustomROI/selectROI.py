# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:37:31 2020

@author: fubar
"""
import os, os.path
import numpy as np

import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


class myRectangleSelector():
    def __init__(self, img, parent_dir_abs):
        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0
        self.img = img
        self.fig = plt.subplots(1,2)
        self.axes = self.fig[0].axes
        self.axes[0].imshow(img)
        self.parent_dir_abs = parent_dir_abs
        
    def onselect(self, eclick, erelease):
        "eclick and erelease are matplotlib events at press and release."
        print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
        print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
        print('used button  : ', eclick.button)
        
        #update the second axes with the selected area
        x0 = int(eclick.xdata)
        y0 = int(eclick.ydata)
        x1 = int(erelease.xdata)
        y1 = int(erelease.ydata)
        img_crop = self.img[y0:y1, x0:x1]
        self.axes[1].clear()
        self.axes[1].imshow(img_crop)
        self.axes[1].figure.canvas.draw()
        
        #update the attributes of the class
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        
        coords = np.array([x0, y0, x1, y1])
        # print(f"self.x0: {self.x0}")
        coords_file = os.path.abspath(os.path.join(self.parent_dir_abs, 'coordinates.txt'))
        np.savetxt(coords_file, coords)
        
    def getCoordinates(self):
        return [self.x0, self.y0, self.x1, self.y1]
        
        
    
def toggle_selector(event):
    print('Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print('RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print('RectangleSelector activated.')
        toggle_selector.RS.set_active(True)


def selectROI(img, parent_dir_abs):

    # print (f"Switched to:{matplotlib.get_backend()}")
    #   gui_backends = mpl.rcsetup.interactive_bk
    # mpl.use('Qt5Agg', force=True)  # set interactive backend 'Qt5Agg'

    selector = myRectangleSelector(img, parent_dir_abs)
    
    toggle_selector.RS = RectangleSelector(selector.axes[0], selector.onselect, drawtype='box')
    

    selector.fig[0].canvas.mpl_connect('key_press_event', toggle_selector)
    selector.fig[0].show()
    input(prompt="Press Q to proceed")
    return selector
    

if __name__ == "__main__":
    img = cv2.imread("wololo.jpg")
    parent_dir_abs = os.curdir
    selectROI(img,parent_dir_abs)
    