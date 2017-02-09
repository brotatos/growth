import cv2
import matplotlib.pyplot as plt
import numpy as np

def read_image(filename):
    img = cv2.imread(filename)
    b, g, r = cv2.split(img)
    return cv2.merge([r, g, b])

rgb_img = read_image('Lid UP cropped plus.png')


rgb_img = cv2.medianBlur(rgb_img, 9)
#rgb_img = cv2.blur(rgb_img, (5, 5))
#rgb_img = cv2.GaussianBlur(rgb_img, (5, 5), 2)
plt.figure()
plt.imshow(rgb_img)

gimg = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
#thresh =\
# cv2.adaptiveThreshold(gimg,
#                       165,
#                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                       cv2.THRESH_BINARY_INV,
#                       11,
#                       2)

res, thresh = cv2.threshold(gimg, 165, 255, cv2.THRESH_BINARY_INV)
plt.figure()
plt.imshow(thresh, cmap='gray')

im2, contours, hierarchy = cv2.findContours(thresh,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
contoured = rgb_img.copy()
cv2.drawContours(contoured, contours, -1, (0, 255, 0), 3)
print(len(contours))


plt.figure()
plt.imshow(contoured)
plt.show()
