import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('myMatplotlibStylesheet.mplstyle')
mpl.rcParams['figure.figsize'] = 6.0,6.0
mpl.rcParams['figure.subplot.bottom'] = 0.1
mpl.rcParams['figure.subplot.left'] = 0.1
mpl.rcParams['figure.subplot.right'] = 0.9
mpl.rcParams['figure.subplot.top'] = 0.9

img = cv2.imread('02_CroppedROI.png',0)
equ = cv2.equalizeHist(img)
cv2.imwrite("03_HistogramEqualized.png",equ)


images = [img, 0,
          equ, 0
		 ]
titles = ['Original','Histogram',
          'Equalized','Equalized histogram'
         ]

for i in np.arange(2):
    plt.subplot(2,2,i*2+1),plt.imshow(images[i*2],'gray')
    plt.title(titles[i*2]), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,i*2+2),plt.hist(images[i*2].ravel(),128)
    plt.title(titles[i*2+1]), plt.xticks([]), plt.yticks([])
plt.show()
plt.savefig("03_HistogramEqualizationEffect.pdf")
plt.savefig("03_HistogramEqualizationEffect.png")
