import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy

#Read in the image given a filename
def read_image (filename):
    img = cv2.imread(filename)
    b, g, r = cv2.split(img)
    return cv2.merge([r, g, b])

#show the image in grayscale
def sg (img, title=''):
    pass
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(img, cmap='gray', interpolation='none')

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


#read in the image
orig = read_image('test.png')
gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray)

cv2.normalize(cl1, cl1, 0, 100, cv2.NORM_MINMAX, cv2.CV_8UC1)
hist = cv2.calcHist([cl1],[0],None,[256],[0,256])
plt.plot(hist)
# getting the max not including the 0 index
max_hist = np.argmax(hist[1:])
# print max_hist

ret,thresh = cv2.threshold(cl1, max_hist + 20, 100, cv2.THRESH_BINARY)
thresh = cv2.medianBlur(thresh, 9)

#use canny edge detection to find the edges
edges = auto_canny(thresh)

# dilate
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=2)

morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

#find the contours
morph_rgb = cv2.cvtColor(morph, cv2.COLOR_GRAY2RGB)
im2, contours, hierarchy = cv2.findContours(morph,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(orig, [contours[5]], -1, (0, 255, 0), 3)
print contours[0]

plt.figure()
plt.imshow(orig, interpolation = 'none')
plt.show()
