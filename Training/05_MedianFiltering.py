import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('myMatplotlibStylesheet.mplstyle')
mpl.rcParams['figure.figsize'] = 6.0,6.0
mpl.rcParams['figure.subplot.bottom'] = 0.1
mpl.rcParams['figure.subplot.left'] = 0.1
mpl.rcParams['figure.subplot.right'] = 0.9
mpl.rcParams['figure.subplot.top'] = 0.9

img = cv2.imread('04_GuassianBlur.png', 0)

#Here is the code to use the median filter:

median_blur= cv2.medianBlur(img,5)
cv2.imwrite("05_MedianFiltered.png",median_blur)


# plot all the images and their histograms
images = [img, 0,
          median_blur, 0]

titles = ['Gaussian blurred','Gaussianized histogram',
          'Median filtered', 'Medianized histogram']

for i in np.arange(2):
    plt.subplot(2,2,i*2+1),plt.imshow(images[i*2],'gray')
    plt.title(titles[i*2]), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,i*2+2),plt.hist(images[i*2].ravel(),128)
    plt.title(titles[i*2+1]), plt.xticks([]), plt.yticks([])

plt.show()
plt.savefig("05_MedianFilteredEffect.png")
plt.savefig("05_MedianFilteredEffect.pdf")
