import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image (assuming it's a colored brain image with skull)
image = cv2.imread('Dataset/Brain6_stripped_and_processed.png')

grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

seed = (200,200)

#Parameters for region growing
neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
region_threshold = 0.2
region_size = 1
intensity_difference = 0
neighbor_points_list = []
neighbor_intensity_list = []

#Mean of the segmented region
region_mean = grey_image[seed]

#Input image parameters
height, width = grey_image.shape
image_size = height * width

#Initialize segmented output image
segmented_image = np.zeros((height, width, 1), np.uint8)

#Region growing until intensity difference becomes greater than certain threshold
while (intensity_difference < region_threshold) & (region_size < image_size):
    print(intensity_difference, region_size)
    print(region_threshold, image_size)
    #Loop through neighbor pixels
    for i in range(4):
        #Compute the neighbor pixel position
        x_new = seed[0] + neighbors[i][0]
        y_new = seed[1] + neighbors[i][1]

        #Boundary Condition - check if the coordinates are inside the image
        check_inside = (x_new >= 0) & (y_new >= 0) & (x_new < height) & (y_new < width)

        #Add neighbor if inside and not already in segmented_image
        if check_inside:
            if segmented_image[x_new, y_new] == 0:
                neighbor_points_list.append([x_new, y_new])
                neighbor_intensity_list.append(grey_image[x_new, y_new])
                segmented_image[x_new, y_new] = 255

    #Add pixel with intensity nearest to the mean to the region
    distance = abs(neighbor_intensity_list-region_mean)
    pixel_distance = min(distance)
    index = np.where(distance == pixel_distance)[0][0]
    segmented_image[seed[0], seed[1]] = 255
    region_size += 1

    #New region mean
    region_mean = (region_mean*region_size + neighbor_intensity_list[index])/(region_size+1)

    #Update the seed value
    seed = neighbor_points_list[index]
    #Remove the value from the neighborhood lists
    neighbor_intensity_list[index] = neighbor_intensity_list[-1]
    neighbor_points_list[index] = neighbor_points_list[-1]

plt.figure()
plt.subplot(1,2,1)
plt.imshow(image)
plt.title('Begin picture')

plt.subplot(1,2,2)
plt.imshow(segmented_image)
plt.title('segment')

plt.show()


