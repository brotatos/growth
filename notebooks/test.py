import cv2
import matplotlib.pyplot as plt
import numpy as np


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def read_image(filename):
    img = cv2.imread(filename)
    b, g, r = cv2.split(img)
    return cv2.merge([r, g, b])


def sg(img, title=''):
    pass
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(img, cmap='gray', interpolation='none')


rgb_img_orig = read_image('Lid UP cropped plus.png')
#rgb_img_orig = read_image('cropped 2.png')
rgb_img = rgb_img_orig.copy()

#canny = cv2.Canny(rgb_img, 55, 255)
#im2, contours, hierarchy = cv2.findContours(canny,
#                                            cv2.RETR_TREE,
#                                            cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(rgb_img_orig, contours, -1, (0, 255, 0), 3)
#plt.figure()
#plt.imshow(rgb_img_orig)
#print contours[0]
#plt.figure()
#plt.imshow(rgb_img_orig)

rgb_img = cv2.medianBlur(rgb_img, 9)
plt.figure()
plt.imshow(rgb_img)
# grayscale
gimg = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

# binary thresh
res, thresh = cv2.threshold(gimg, 165, 255, cv2.THRESH_BINARY)
sg(thresh, title='thresh')

# morphology - removing super small holes
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
sg(opening, title='morph')

# sure background
sure_bg = cv2.dilate(opening, kernel, iterations=3)
sg(sure_bg, title='dilate')

#finding sure fg area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
sg(dist_transform, title='distance transform')

## sure_fg threshold
ret, sure_fg = cv2.threshold(dist_transform,
                             0.1 * dist_transform.max(),
                             #0.7 * dist_transform.max(),
                             255,
                             cv2.THRESH_BINARY)
#print dist_transform.max()
sg(sure_fg, title='threshold')

# Find unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
#ret, markers = cv2.connectedComponents(sure_fg)
output = cv2.connectedComponentsWithStats(sure_fg)
num_labels, markers, stats, centroids = output

# get areas of contours
values = [0] * num_labels

for row in range(markers.shape[0]):
    for col in range(markers.shape[1]):
        values[markers[row, col]] += 1

del values[0]

print values
plt.figure()
#plt.hist(range(len(values)), values)
plt.hist(values)

print "mean", np.mean(values)
print "median", np.median(values)
print "max", np.max(values)
print "min", np.min(values)
print "std dev", np.std(values)

#print np.index(np.min(values))
# Add one to all labels so sure background is not 0, but 1
markers = markers + 1
markers[unknown==255] = 0

## watershed
markers = cv2.watershed(rgb_img_orig, markers)
rgb_img_orig[markers == -1] = [0, 0, 255]
plt.figure()
plt.imshow(rgb_img_orig, interpolation='none')


plt.show()
