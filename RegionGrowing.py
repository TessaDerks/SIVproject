import cv2
import numpy as np

def region_growing(image, seed_point, threshold):
    height, width = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=bool)

    # Get the seed pixel intensity
    seed_intensity = image[seed_point[1], seed_point[0]]

    # Initialize the region-growing stack
    stack = [seed_point]

    while stack:
        x, y = stack.pop()

        if visited[y, x]:
            continue

        visited[y, x] = True

        # Check the intensity difference
        if abs(int(image[y, x]) - int(seed_intensity)) <= threshold:
            segmented[y, x] = 255

            # Add neighboring pixels to the stack
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy

                if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                    stack.append((nx, ny))

    return segmented

def findRegions(image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # find region based on seed choice
    white = region_growing(grey_image, (167,390),55)
    grey = region_growing(grey_image,(150,192) ,25)
    fluid = region_growing(grey_image, (232,243),155)
    return grey, white, fluid




