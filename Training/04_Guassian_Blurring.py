import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('myMatplotlibStylesheet.mplstyle')
mpl.rcParams['figure.figsize'] = 6.0,6.0
mpl.rcParams['figure.subplot.bottom'] = 0.1
mpl.rcParams['figure.subplot.left'] = 0.1
mpl.rcParams['figure.subplot.right'] = 0.9
mpl.rcParams['figure.subplot.top'] = 0.9

img = cv2.imread('03_HistogramEqualized.png',0)

blur = cv2.GaussianBlur(img,(5,5),0)
cv2.imwrite('04_GuassianBlur.png',blur)

# plot all the images and their histograms
images = [img, 0,
          blur, 0]

titles = ['Equalized','Histogram',
          'Gaussian blurred','Gaussianized histogram']

for i in np.arange(2):
    plt.subplot(2,2,i*2+1),plt.imshow(images[i*2],'gray')
    plt.title(titles[i*2]), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,i*2+2),plt.hist(images[i*2].ravel(),128)
    plt.title(titles[i*2+1]), plt.xticks([]), plt.yticks([])

plt.show()
plt.savefig("04_GaussianBlurringEffect.png")
plt.savefig("04_GaussianBlurringEffect.pdf")
