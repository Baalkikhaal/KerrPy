import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('myMatplotlibStylesheet.mplstyle')
mpl.rcParams['figure.figsize'] = 4.0,2.0
mpl.rcParams['figure.subplot.bottom'] = 0.1
mpl.rcParams['figure.subplot.left'] = 0.1
mpl.rcParams['figure.subplot.right'] = 0.9
mpl.rcParams['figure.subplot.top'] = 0.9

img = cv2.imread('05_MedianFiltered.png',0)

# Otsu's thresholding
ret, otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imwrite('06_BinaryOtsu.png', otsu)

# plot all the images

images = [img, otsu]

titles = ['Median filtered', 'Otsu Binarized']

for i in np.arange(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])


plt.show()
plt.savefig("06_BinaryOtsuEffect.pdf")
plt.savefig("06_BinaryOtsuEffect.png")

