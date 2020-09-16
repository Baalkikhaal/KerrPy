# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:52:16 2020

@author: fubar
"""

import os, os.path

import cv2

import numpy as np

from KerrPy.Image.routinesOpenCV import readImage, cropImageDetails, cropImage, writeImage


from KerrPy.Image.processROI import findAdaptiveROI, restoreColorROI

from KerrPy.Figure.routinesMatplotLib import displayHistogram, displayMorphTrans, displaySubPlots, saveFig

from KerrPy.Image.processOpenCV import fitEllipse

# save the current directory before relocating
cur_dir = os.path.abspath(os.curdir)

training_folder = 'DataTraining'

os.chdir(training_folder)

test_big_domain_folder = 'MediumDomain'

os.chdir(test_big_domain_folder)

img_test_file = 'testMediumDomain.png'

pulse_index = 2
iter_index = 0
exp_index = 0
parent_dir_abs = os.path.abspath(os.curdir)



# read the image

img = readImage(img_test_file)

# remove the bottom strip

img_crop = cropImageDetails(img)

writeImage("00_Cropped.png", img_crop)

# find the adaptive ROI


adaptive_ROI = findAdaptiveROI(pulse_index, iter_index, exp_index, img, parent_dir_abs) 

# crop the image with adaptive ROI

img_crop_ROI = cropImage(img, adaptive_ROI)

writeImage("01_CroppedROI.png", img_crop_ROI)

# histogram equalization

img_equ = cv2.equalizeHist(img_crop_ROI)


writeImage("03_HistogramEqualized.png", img_equ)

# compare the img_equ with img_crop_ROI
images_equ = [img_crop_ROI, 0,
           img_equ, 0
 		 ]
titles_equ = ['Original','Histogram',
           'Equalized','Equalized histogram'
          ]

fig_equ = displayHistogram(images_equ, titles_equ)


saveFig("03_HistogramEqualizationEffect.png", fig_equ)

# gaussian blur
img_blur = cv2.GaussianBlur(img_equ,(5,5),0)

writeImage("04_GuassianBlur.png", img_blur)

# compare the img_blur with img_equ

images_blur = [img_equ, 0,
               img_blur, 0]

titles_blur = ['Equalized','Histogram',
          'Gaussian blurred','Gaussianized histogram']

fig_blur = displayHistogram(images_blur, titles_blur)

saveFig("04_GaussianBlurringEffect.png", fig_blur)

# median filtering

img_med = cv2.medianBlur(img_blur,5)

writeImage("05_MedianFiltered.png", img_med)

images_med = [img_blur, 0,
          img_med, 0]

# compare the img_med with img_blur

titles_med = ['Gaussian blurred','Gaussianized histogram',
          'Median filtered', 'Medianized histogram']

fig_med = displayHistogram(images_med, titles_med)

saveFig("05_MedianFilteredEffect.png", fig_med)

# otsu thresholding

ret, img_otsu = cv2.threshold(img_med, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

writeImage("06_BinaryOtsu.png", img_otsu)

# compare img_otsu with img_med

images_otsu = [img_med, 0,
               img_otsu, 0]

titles_otsu = ['Median filtered', 'Medianized histogram',
          'Otsu Binarized', 'Otsu histogram']

fig_otsu = displayHistogram(images_otsu, titles_otsu)

saveFig("06_BinaryOtsuEffect.png", fig_otsu)

# morphological transformation

kernel_level_one = 9
kernel = np.ones((kernel_level_one, kernel_level_one),np.uint8)

img_opened = cv2.morphologyEx(img_otsu, cv2.MORPH_OPEN, kernel)

img_closed = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)


img_openedthenclosed = cv2.morphologyEx(img_opened, cv2.MORPH_CLOSE, kernel)

kernel_level_two = 15

kernel = np.ones((kernel_level_two, kernel_level_two),np.uint8)

img_openthenclosedthenopened = cv2.morphologyEx(img_openedthenclosed, cv2.MORPH_OPEN, kernel)

img_openthenclosedthenopenedthenclosed = cv2.morphologyEx(img_openthenclosedthenopened, cv2.MORPH_CLOSE, kernel)

writeImage("07_RemoveSpeckles_ErosionAndDilation.png", img_openthenclosedthenopenedthenclosed)

title_o = f'O({kernel_level_one})'
title_c = f'C({kernel_level_one})'
title_oc = f'O({kernel_level_one})C({kernel_level_one})'
title_oco = f'O({kernel_level_one})C({kernel_level_one})O({kernel_level_two})'
title_ococ = f'O({kernel_level_one})C({kernel_level_one})O({kernel_level_two})C({kernel_level_two})'

titles_morph = [ ['Otsu', title_o, title_c ],
                [title_oc, title_oco, title_ococ ]
                ]

images_morph = [ [img_otsu, img_opened, img_closed ],
                [ img_openedthenclosed , img_openthenclosedthenopened, img_openthenclosedthenopenedthenclosed]
                ]

fig_morph = displayMorphTrans(images_morph, titles_morph)

saveFig("07_ErosionAndDilationEffect.png", fig_morph)

# Canny edge detection

img_speckless = img_openthenclosedthenopenedthenclosed

img_edges = cv2.Canny(img_speckless,100,200)

writeImage("08_CannyEdgeEffect.png", img_speckless)

#compare img_edges with img_speckless

images_edges = [ img_speckless, img_edges ]

titles_edges = [ 'Without speckles', 'Canny edges']

fig_edges = displaySubPlots(images_edges, titles_edges)

saveFig("08_CannyEdgeEffect.png", fig_edges )

# fit the ellipse

pulse, img_color = fitEllipse(img_edges, adaptive_ROI)

print('{}{}'.format('pulse:', pulse))

writeImage("09_FitEllipse.png", img_color)

# restore the ROI onto the original image

img_color_restored = restoreColorROI(img_color, img, adaptive_ROI)

writeImage("10_restoredROI.png", img_color_restored)



# restore the path to parent dir
os.chdir(cur_dir)

