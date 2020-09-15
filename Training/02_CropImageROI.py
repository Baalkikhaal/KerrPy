import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ScanThroughCrops import FindAdaptiveROI, CropSequenceGenerate, CropImage





if __name__ == '__main__':
    nucleation_down =   1 # 0 for nucleation up
    center_ROI      =   (511,672)  #center of the object to be identified
    aspr_ROI        =   2/3   # x_width/y_width for ROI. This is found by TRAINING
    displayImages   =   True
    DEBUG           =   True
    
    img = cv2.imread('img4.png',0)
    
    mpl.style.use('myMatplotlibStylesheet.mplstyle')
    mpl.rcParams['figure.figsize'] = 4.0,2.0
    mpl.rcParams['figure.subplot.bottom'] = 0.1
    mpl.rcParams['figure.subplot.left'] = 0.1
    mpl.rcParams['figure.subplot.right'] = 0.90
    mpl.rcParams['figure.subplot.top'] = 0.90


    mpl.rcParams['figure.max_open_warning'] = 40
    
    ROI = (1022,1344)
    cropsequence = CropSequenceGenerate(img,(center_ROI,ROI))
    cropimg = CropImage(img,cropsequence, displayImages)
    cv2.imwrite("02_CroppedROI.png", cropimg)
