import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('myMatplotlibStylesheet.mplstyle')
mpl.rcParams['figure.figsize'] = 4.0,2.0
mpl.rcParams['figure.subplot.bottom'] = 0.1
mpl.rcParams['figure.subplot.left'] = 0.1
mpl.rcParams['figure.subplot.right'] = 0.90
mpl.rcParams['figure.subplot.top'] = 0.90

img = cv2.imread('07_RemoveSpeckles_ErosionAndDilation.png',0)

#find the Canny edges
edges = cv2.Canny(img,100,200)
cv2.imwrite("08_CannyEdges.png",edges)
np.save("08_CannyEdges.npy",edges)

# plot all the images

images = [img, edges]

titles = ['Without speckles', 'Canny edges']

for i in np.arange(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])


plt.show()
plt.savefig("08_CannyEdgeEffect.pdf")
plt.savefig("08_CannyEdgeEffect.png")

