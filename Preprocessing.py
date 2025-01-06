import cv2
import numpy as np
import matplotlib.pyplot as plt
import Main


def filter_image(image):
    # Resize the image to streamline process
    resized_image = cv2.resize(image, (500, 600), interpolation=cv2.INTER_CUBIC)
    # Transform image into greyscale
    grey_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # reduce noise of image while perserving edges with median filter
    filtered_image = cv2.medianBlur(grey_image, 5)

    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)

    return filtered_image

def skull_stripping(image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh_image = cv2.threshold(grey_image,75,255,cv2.THRESH_OTSU)
    ret, markers = cv2.connectedComponents(thresh_image)

    #Get the area taken by each component. Ignore label 0 since this is the background.
    marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
    #Get label of largest component by area
    largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above
                         
    #Get pixels which correspond to the brain
    brain_mask = markers==largest_component

    brain_image = image.copy()
    #In a copy of the original image, clear those pixels that don't correspond to the brain
    brain_image[brain_mask==False] = (0,0,0)

    return brain_image


def findBrainContour(src, segmented_image):

    brain = src.copy()
    result = segmented_image.copy()

    # Extract all the external point top/bottom approach and fill the columns of pixels of white
    for x in range(0, brain.shape[1]):

        if np.all(brain[:, x] == 0):
            continue
        
        non_zero = cv2.findNonZero(brain[:, x])
    
        y_top = non_zero[0][0][1]
        y_bottom = non_zero[-1][0][1]

        brain[y_top-5:y_bottom+5, x] = (255, 255, 255)

    # Extract all the external point left/right approach and fill the columns of pixels of white
    for y in range(0, brain.shape[0]):

        if np.all(brain[y, :] == 0):
            continue
        
        non_zero = cv2.findNonZero(brain[y, :])

        x_left = non_zero[0][0][1]
        x_right = non_zero[-1][0][1]

        brain[y, x_left-5:x_right+5] = (255, 255, 255)

    # Create and apply mask
    gray = cv2.cvtColor(brain, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(brain.shape, dtype=np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    cv2.drawContours(mask, contours, -1, (255, 255, 255), cv2.FILLED)

    result[mask == False] = (0, 0, 0)

    return result





