# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 21:41:20 2020

@author: fubar
"""

import sys

from PyQt5.QtWidgets import QApplication

from KerrPy.Fits.processData import processDataWithWindow

from KerrPy.Window.processWindow import initializeWindow



# controls, space = processDataWithWindow(static_ax, dynamic_ax)

if __name__ == '__main__':
    # appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext
   
    app = QApplication([])
    # create a Widget to hold all the upcoming subwidgets
    
    static_ax, dynamic_ax = initializeWindow()
    
    # exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
    
    exit_code = app.exec_()
    
    sys.exit(exit_code)
