import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_image(image_path):

    # read the image
    image = cv2.imread(image_path) 
    return image

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
    # Copy the image
    #image = src.copy()

    # Convert to grayscale
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image using OTSU method. In this way it is possible to use the optimal threshold value based on the image content/histogram
    _, thresh = cv2.threshold(src=grey_image, thresh=0, maxval=255, type=cv2.THRESH_OTSU)

    # Create mask (all black for now)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[thresh != 0] = (0, 0, 255)

    # Extract areas inside the image
    _, markers = cv2.connectedComponents(thresh)

    # Calculate the size of each area
    marker_areas = [np.sum(markers == m) for m in range(np.max(markers)) if m != 0]

    # Sort each marker by area
    sorted_markers = sorted(enumerate(marker_areas, start=1), key=lambda x: x[1], reverse=True)

    # Create a mask
    mask = np.zeros_like(image)

    # Add each area to the mask except for the first (skull) and the smaller ones
    for marker, area in sorted_markers:

        if marker == 1:
            continue

        if area < 4000:
            continue

        mask[markers == marker] = (255, 255, 255)

    # Apply the mask to the original image
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    stripped_image = image.copy()
    stripped_image[mask == False] = (0, 0, 0)

    mask = np.zeros_like(image)
    mask[markers == 1] = (255, 255, 255)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    skull = image.copy()
    skull[mask == False] = (0, 0, 0)

    # If the image has no skull, the brain will be all black
    # Check this and in that case return the original image and the skull (all black)
    if not (stripped_image > 0).any():
        stripped_image, skull = img, brain
    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)

    return stripped_image

def skull_stripping2(image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh_image = cv2.threshold(grey_image,100,255,cv2.THRESH_OTSU)
    
    plt.figure(figsize=(8, 5))
    subplot_images(image, 1, 'original')
    subplot_images(thresh_image,2, 'applying otsu')
 
    colormask = np.zeros(image.shape, dtype=np.uint8)
    colormask[thresh_image!=0] = np.array((0,0,255))
    blended_image = cv2.addWeighted(image,0.7,colormask,0.1,0)

    subplot_images(blended_image, 3, 'blended image')

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

    subplot_images(brain_image, 4, 'skull stripped image')

    plt.show()

    return brain_image

#make subplot for max 9 images
def subplot_images(image, nr, title):
    plt.subplot(3, 3, nr)
    plt.imshow(image)
    plt.title(title)

img = get_image('Dataset/Brain6.png')
# Show the original and segmented images
#plt.figure(figsize=(8, 5))
 
filtered_img = filter_image(img)
#stripped_image = skull_stripping(filtered_image)
stripped_img = skull_stripping2(filtered_img)
#stripped_img = skull_stripping2(img)

#cv2.imwrite('Dataset/Brain6_stripped_and_processed.png', stripped_img)

#plt.show()


