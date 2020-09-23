# from fbs_runtime.application_context.PyQt5 import ApplicationContext

# from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWidgets import QHBoxLayout

from matplotlib.backends.qt_compat import is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

from matplotlib.figure import Figure

import cv2

import sys

import os, os.path


# event handlers here

def updateStaticCanvas(static_ax, img):
    ''' Once the Line Edit is returned, read the image and
        draw on the FigureCanvas()
    '''
        
    
    # clear the current image
    static_ax.clear()
    
    # buffer the update image
    static_ax.imshow(img)
    
    # draw the image on the Qt backend FigureCanvas
    static_ax.figure.canvas.draw()
    
    return static_ax
    
def updateDynamicCanvas(dynamic_ax, img_restored):
    
    # clear the current image
    dynamic_ax.clear()
    
    # buffer the updated image
    dynamic_ax.imshow(img_restored, cmap = 'Greys')
    
    # draw the image on the Qt backend for matplotlib i.e. FigureCanvas
    
    dynamic_ax.figure.canvas.draw()
    
    return dynamic_ax

def initializeWindow():
    """
        0. Initialize the window to hold the raw image
            and restored image
        1. Return the window
    """
    myWindow = QWidget()
    myWindow.setWindowTitle("Custom ROI selection for Canny Edge Detection")
    
    ### create the layout.
    
    ###     Design is two horizontal layout widgets.
    
    ###     The left one hosts the inputs.
    
    ###     The right one hosts the outpus.
    
    layout  = QHBoxLayout()
    
    ### create the Figure widgets
    
    static_canvas   = FigureCanvas(Figure(figsize=(5, 5)))
    static_ax       = static_canvas.figure.subplots()
    
    # we will draw the canvas later; after defining the drawing functions
    
    dynamic_canvas  = FigureCanvas(Figure(figsize=(5, 5)))
    dynamic_ax      = dynamic_canvas.figure.subplots()
    
    # draw static and dynamic figures
    
    img_display_file = "wololo.jpg"
    
    # read the image 
    img_display = cv2.imread(img_display_file, cv2.IMREAD_GRAYSCALE)
    
    updateStaticCanvas(static_ax, img_display)
    updateDynamicCanvas(dynamic_ax, img_display)
    
    ### add the widgets to the horizontal layout
    
    layout.addWidget(static_canvas)
    layout.addWidget(dynamic_canvas)

    # now that the layout is created, set the layout to the main window widget
    
    myWindow.setLayout(layout)
    
    #    
    # define the window signals
    #
    # roiSelected.returnPressed.connect(updateStaticCanvas)



    return static_ax, dynamic_ax

def processWindow():
    
    app = QApplication([])
    # create a Widget to hold all the upcoming subwidgets
    
    static_ax, dynamic_ax = initializeWindow()

    # exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
    
    exit_code = app.exec_()
    
    sys.exit(exit_code)
    
    return static_ax, dynamic_ax


if __name__ == '__main__':
    # appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext
   
    app = QApplication([])
    # create a Widget to hold all the upcoming subwidgets
    
    myWindow, static_ax, dynamic_ax = initializeWindow()
    
    #
    # show the app draw default image
    #
    myWindow.show()
    
    # exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
    
    exit_code = app.exec_()
    
    sys.exit(exit_code)
