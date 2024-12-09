import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image (assuming it's a colored brain image with skull)
image = cv2.imread('Dataset/Brain2.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Threshold Processing
ret, bin_img = cv2.threshold(gray_image,
							0, 255, 
							cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# noise removal
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


# Apply Gaussian Blur to reduce noise and improve segmentation
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
#bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN,kernel,iterations=2)

# Apply Otsu's thresholding to create a binary image
_, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Perform morphological operations to close gaps in the foreground and background
kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)



plt.figure(figsize=(10, 10))
# sure background area
sure_bg = cv2.dilate(bin_img, kernel, iterations=5)

# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(sure_bg)
plt.title("Sure Background")

# Distance transform
dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 3)

#foreground area
ret, sure_fg = cv2.threshold(dist, 0.2 * dist.max(), 255, cv2.THRESH_BINARY)
sure_fg = sure_fg.astype(np.uint8) 
plt.subplot(1,2,2)
plt.imshow(sure_fg)
plt.title('Sure Foreground')

# unknown area
unknown = cv2.subtract(sure_bg, sure_fg)


plt.show()


