# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 14:08:56 2020

@author: fubar
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 00:03:18 2020

@author: fubar
"""

import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

def onselect(eclick, erelease):
    "eclick and erelease are matplotlib events at press and release."
    # print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
    # print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
    # print('used button  : ', eclick.button)
    
    # save the state to function attributes
    span = (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)
    onselect.span = span
    print('hello')

def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and myWidget.get_active():
        print('EllipseSelector deactivated.')
        myWidget.set_active(False)
    if event.key in ['A', 'a'] and not myWidget.get_active():
        print('EllipseSelector activated.')
        myWidget.set_active(True)
    if event.key in ['N', 'n'] and myWidget.get_active():
        print('Next image.')
        updateAxes.counter += 1
        printSpan()
        updateAxes()

def printSpan():
    print(onselect.span)
    
def updateAxes():
    mode = updateAxes.counter
    x = np.arange(100.) / 99
    y = np.sin(mode * x)
    # myWidget.ax.clear()
    myWidget.ax.plot(x,y)
    myWidget.update()
    
    # reconnect the default events and event handlers
    # myWidget.connect_default_events()
    
x = np.arange(100.) / 99
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)

# a reference to the RectangleSelector widget is needed
# to prevent from Garbage Collection
myWidget = RectangleSelector(ax, onselect)

#initialize the counter for the curves
updateAxes.counter = 0

#initialize the span coordinates defining the state of the Rectangle Selection
onselect.span = (0., 0., 1., 1.)
myWidget.connect_event('key_press_event', toggle_selector)

plt.show()