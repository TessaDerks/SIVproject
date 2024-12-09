import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image (assuming it's a colored brain image with skull)
image = cv2.imread('Dataset/Brain6_stripped.png')

grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise and improve segmentation
#blurred_image = cv2.GaussianBlur(grey_image, (5, 5), 0)
#bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN,kernel,iterations=2)

# Apply Otsu's thresholding to create a binary image
_, binary_image = cv2.threshold(grey_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Perform morphological operations to close gaps in the foreground and background
kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)
opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations =2)

plt.figure(figsize=(8, 8))
# sure background area
#sure_bg = cv2.dilate(binary_image, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations =3)
sure_bg_ = cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2BGR)

# Plot the original image
plt.subplot(2, 2, 1)
plt.imshow(sure_bg_)
plt.title("Sure Background")

# Distance transform
#dist = cv2.distanceTransform(binary_image, cv2.DIST_L2, 3)
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

#foreground area
#ret, sure_fg = cv2.threshold(dist, 0.7 * dist.max(), 255, cv2.THRESH_BINARY)
ret, sure_fg = cv2.threshold(dist, 0.2 * dist.max(), 255, 0)
sure_fg = sure_fg.astype(np.uint8) 
sure_fg_ = cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2BGR)
plt.subplot(2,2,2)
plt.imshow(sure_fg_)
plt.title('Sure Foreground')

# unknown area
unknown = cv2.subtract(sure_bg, sure_fg)
unknown_ = cv2.cvtColor(unknown, cv2.COLOR_GRAY2BGR)
plt.subplot(2,2,3)
plt.imshow(unknown_)
plt.title('unknown')

ret, markers = cv2.connectedComponents(unknown)
markers = markers +1
markers[unknown==255] = 0
markers = cv2.watershed(image,markers)
print(np.unique(markers))
image[markers == 0] = [255,0,0]

plt.subplot(2,2,4)
plt.imshow(image)
plt.title('end result')
plt.show()


