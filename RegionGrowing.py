import cv2
import numpy as np
import matplotlib.pyplot as plt
import Main


def region_growing2(image, seed, threshold):


    #Parameters for region growing
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    region_threshold = 0.2
    region_size = 1
    intensity_difference = 0
    neighbor_points_list = []
    neighbor_intensity_list = []

    #Mean of the segmented region
    region_mean = image[seed]

    #Input image parameters
    height, width = image.shape
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
                    neighbor_intensity_list.append(image[x_new, y_new])
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


def region_growing(image, seed, threshold):
    image = image.astype(np.int32)
    rows, cols = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=bool)
    
    stack = [seed]
    intensity = image[seed]
    while stack:
        x, y = stack.pop()
        if not visited[x, y] and abs(intensity - image[x, y]) < threshold:
            segmented[x, y] = 255
            visited[x, y] = True
            # Add neighbors to the stack
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    stack.append((nx, ny))

    segmented = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
    return segmented

def findRegions(image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plot a seed in case you wanna check
    image_with_seed = cv2.circle(image, (310,491), radius=5, color=(255, 0, 0), thickness=-1) 
    Main.subplot_images(image_with_seed, 3, 'image with seed')
    # find region based on seed choice
    grey = region_growing(grey_image, (310,491),15)
    white = region_growing(grey_image, (150,400),5)
    #white = region_growing(grey_image, (325,370),5)
    fluid = region_growing(grey_image, (250,250),25)

    return grey, white, fluid




