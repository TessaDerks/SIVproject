import cv2
import numpy as np
import matplotlib.pyplot as plt

# read the image
image = cv2.imread('Dataset/Brain5.jpg') 

# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = image.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)

# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# number of clusters (K)
k = 3
compactness, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


# convert back to 8 bit values
centers = np.uint8(centers)


# flatten the labels array
labels = labels.flatten()

#mask = cv2.inRange((255,0,255), (255,128,0), (153,104,255))
custom_colors = np.array([[255, 0, 255],  # Red
                          [255, 128, 0],  # Green
                          [153, 104, 255]])

centers = custom_colors


# convert all pixels to the color of the centroids
#segmented_image = centers[labels]

# Choose the cluster you want to display (for example, cluster 1)
cluster_to_display = 0  # Change this to 0, 1, or 2 to show different clusters

# Create a mask for the chosen cluster
mask = (labels == cluster_to_display)

# Reshape the mask to match the original image shape (height x width)
mask = mask.reshape(image.shape[0], image.shape[1])

# Create an empty image with black background
segmented_cluster_image = np.zeros_like(image)

# Set the pixels corresponding to the selected cluster to the cluster's color'
segmented_cluster_image[mask] = centers[cluster_to_display]

# Show the original and segmented images
plt.figure(figsize=(10, 5))

# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")

# Plot the image showing only the selected segment
plt.subplot(1, 2, 2)
plt.imshow(segmented_cluster_image)
plt.title(f"Cluster {cluster_to_display} Only")

plt.show()

#cv2.imwrite('Results/Brain2_whitematter_unprocessed.png', segmented_cluster_image)

'''
# reshape back to the original image dimension
segmented_image = segmented_image.reshape(image.shape)
print(centers)
# show the image
plt.imshow(segmented_image)
plt.show()

# disable only the cluster number 2 (turn the pixel into black)
masked_image = np.copy(image)
# convert to the shape of a vector of pixel values
masked_image = masked_image.reshape((-1, 3))
# color (i.e cluster) to disable
cluster = 2
masked_image[labels == cluster] = [0, 0, 0]

# convert back to original shape
masked_image = masked_image.reshape(image.shape)
# show the image
plt.imshow(masked_image)
plt.show()
'''