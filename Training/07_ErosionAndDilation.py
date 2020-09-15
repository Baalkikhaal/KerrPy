import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('myMatplotlibStylesheet.mplstyle')
mpl.rcParams['figure.figsize'] = 6.0,6.0
mpl.rcParams['figure.subplot.bottom'] = 0.05
mpl.rcParams['figure.subplot.left'] = 0.05
mpl.rcParams['figure.subplot.right'] = 0.95
mpl.rcParams['figure.subplot.top'] = 0.95


#img = cv2.imread('05_RemoveSpeckles_ErosionAndDilationFirstIteration.png', 0)

img = cv2.imread('06_BinaryOtsu.png', 0)

# morphological transformation i.e. erosion and dilation of the binary image to remove speckles

kernel = np.ones((9,9),np.uint8)

    
# open the image and save
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imwrite('07_RemoveSpeckles_Erosion.png', opened)

# close the image and save
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('07_RemoveSpeckles_Dilation.png', closed)

# morphological transformation (two iterations; second iteration with bigger kernel 15) and save
openedthenclosed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((15,15),np.uint8)
openthenclosedthenopened = cv2.morphologyEx(openedthenclosed, cv2.MORPH_OPEN, kernel)
openthenclosedthenopenedthenclosed = cv2.morphologyEx(openthenclosedthenopened, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('07_RemoveSpeckles_ErosionAndDilation.png', openthenclosedthenopenedthenclosed)

titles = ['Otsu Binarized', 'Opened (erosion) ', 'Closed (dilation)', 'Without speckles']
images = [img, opened, closed, openthenclosedthenopenedthenclosed]

plt.subplot(3,3,2), plt.imshow(images[0],'gray'), plt.title(titles[0]), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,4), plt.imshow(images[1],'gray'), plt.title(titles[1]), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,6), plt.imshow(images[2],'gray'), plt.title(titles[2]), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,8), plt.imshow(images[3],'gray'), plt.title(titles[3]), plt.xticks([]), plt.yticks([])

plt.show()
plt.savefig('07_ErosionAndDilationEffect.pdf')
plt.savefig('07_ErosionAndDilationEffect.png')
