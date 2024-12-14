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
    #filtered_image =  cv2.GaussianBlur(grey_image,(5,5),0)

    # equalize image to improve quality
    #equalized_image = cv2.equalizeHist(filtered_image)
    #equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

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




