import cv2
import numpy as np
import matplotlib.pyplot as plt
import Main


def filter_image(image):
    #plt.figure(figsize=(8, 8))
    #subplot_images(image, 1, 'original')
    # Resize the image to streamline process
    resized_image = cv2.resize(image, (500, 600), interpolation=cv2.INTER_CUBIC)
    # Transform image into greyscale
    grey_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    #subplot_images(grey_image, 2, 'resized and greyed')

    # reduce noise of image while perserving edges with median filter
    filtered_image = cv2.medianBlur(grey_image, 3)
    
    #subplot_images(filtered_image, 3, 'filtered')
    # equalize image to improve quality
    equalized_image = cv2.equalizeHist(filtered_image)
    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
    #subplot_images(equalized_image, 4, 'equalized')

        # Extract the brain
    _, thresh = cv2.threshold(src=filtered_image, thresh=75, maxval=255, type=cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    #subplot_images(thresh, 5, 'threshold pic')

    # Create a black image (all zeros) with the same size as image
    preprocessed_image = np.zeros_like(resized_image)

    # Draw the contour of the brain on merged
    cv2.drawContours(preprocessed_image, [contours[0]], 0, (255, 255, 255), thickness=cv2.FILLED)
    #subplot_images(preprocessed_image.copy(), 6, 'contours')

    # Create a mask for the region inside the contour
    mask = np.zeros_like(filtered_image)
    cv2.drawContours(mask, [contours[0]], 0, (255, 255, 255), thickness=cv2.FILLED)

    # Assign equalized inside the brain
    preprocessed_image[mask == 255] = equalized_image[mask == 255]
    #subplot_images(preprocessed_image.copy(), 7, 'equalized mask')
    # Assign filtered_image outside the brain. 
    # In this way we can denoise the background and make it all black, without doing equalization on it
    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
    preprocessed_image[mask == 0] = filtered_image[mask == 0]
    #subplot_images(preprocessed_image.copy(), 8, 'filtered masked')
    #plt.show()
    return preprocessed_image

def skull_stripping(image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh_image = cv2.threshold(grey_image,100,255,cv2.THRESH_OTSU)
 
    colormask = np.zeros(image.shape, dtype=np.uint8)
    colormask[thresh_image!=0] = np.array((0,0,255))
    blended_image = cv2.addWeighted(image,0.7,colormask,0.1,0)

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




