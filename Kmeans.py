import cv2
import numpy as np
import matplotlib.pyplot as plt
import Main

def kmeans(image):

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

    custom_colors = np.array([[255, 0, 255],  # Red
                            [255, 128, 0],  # Green
                            [153, 104, 255]])

    centers = custom_colors
    # convert all pixels to the color of the centroids
    grey = getSegment(0,labels,image,centers)
    white = getSegment(1,labels,image,centers)
    fluid = getSegment(2,labels,image,centers)
    return grey, white, fluid

def getSegment(cluster_to_display,labels, image, centers):  
    # Create a mask for the chosen cluster
    mask = (labels == cluster_to_display)

    # Reshape the mask to match the original image shape (height x width)
    mask = mask.reshape(image.shape[0], image.shape[1])

    # Create an empty image with black background
    segmented_cluster_image = np.zeros_like(image)

    # Set the pixels corresponding to the selected cluster to the cluster's color'
    segmented_cluster_image[mask] = centers[cluster_to_display]

    return segmented_cluster_image

    