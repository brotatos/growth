# contouring
im2, contours, hierarchy = cv2.findContours(opening,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(rgb_img_orig, contours, -1, (0, 255, 0), 3)
plt.figure()
plt.imshow(rgb_img_orig)
print contours[0]
plt.figure()
plt.imshow(rgb_img_orig)
# actual garbage
#rgb_img = cv2.blur(rgb_img, (5, 5))
#rgb_img = cv2.GaussianBlur(rgb_img, (5, 5), 2)
#thresh =\
# cv2.adaptiveThreshold(gimg,
#                       165,
#                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                       cv2.THRESH_BINARY_INV,
#                       11,
#                       2)


