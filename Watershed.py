import cv2
import numpy as np
import matplotlib.pyplot as plt
import Main

# Read the image (assuming it's a colored brain image with skull)
#image = cv2.imread('Dataset/Brain6_stripped_and_processed.png')
def watershed(image):

    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #grey_image = cv2.GaussianBlur(grey_image, (5,5),0)
    #plt.figure(figsize=(8, 8))
    # Apply Otsu's thresholding to create a binary image
    _, binary_image = cv2.threshold(grey_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #plt.subplot(2, 3, 1)
    #plt.imshow(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR))
    #plt.title("binary image")

    # Perform morphological operations to close gaps in the foreground and background
    kernel = np.ones((3, 3), np.uint8)
    #closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations =2)
    #dilation = cv2.morphologyEx(binary_image,cv2.MORPH_DILATE, kernel)

    # sure background area
    #sure_bg = cv2.dilate(binary_image, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations =2)
    #sure_bg_ = cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2BGR)

    # Plot the original image
    #plt.subplot(2, 3, 2)
    #plt.imshow(sure_bg)
    #plt.title("Sure Background")

    # Distance transform
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    #dist = cv2.distanceTransform(dilation,cv2.DIST_L2,5)

    #foreground area
    #ret, sure_fg = cv2.threshold(dist, 0.05 * dist.max(), 255, 0)
    #ret, sure_fg = cv2.threshold(dist, 0.2 * dist.max(), 128, 0)
    ret, sure_fg = cv2.threshold(dist, 15, 255, cv2.THRESH_BINARY)
    #_, distThresh = cv2.threshold(dist,15,250,cv2.THRESH_BINARY)

    sure_fg = np.uint8(sure_fg)
    #sure_fg_ = cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2BGR)
    #distThresh = distThresh.astype(np.uint8)

    #_, labels = cv2.connectedComponents(distThresh)
    #plt.subplot(2,3,3)
    #plt.imshow(sure_fg)
    #plt.title('Sure Foreground')
    #labels = labels.astype(np.int32)

    #image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # unknown area
    unknown = cv2.subtract(sure_bg, sure_fg)
    unknown2 = cv2.subtract(sure_fg, sure_bg)  ##try some other, addition??
    grey_matter = cv2.subtract(sure_bg, unknown2)

    #unknown_ = cv2.cvtColor(unknown, cv2.COLOR_GRAY2BGR)
    #markers = cv2.watershed(image, labels)
    #plt.subplot(2,3,4)
    #plt.imshow(grey_matter)
    #plt.title('grey matter')

    ret, markers = cv2.connectedComponents(sure_fg) ######
    markers = markers +1
    markers[unknown==255] = 0
  
    markers = cv2.watershed(image,markers)

    print(np.unique(markers))
    #image[markers == -1] = [255,0,0]
    img = image.copy()
    img[markers==-1] = [255,0,0]
    #segmented_image = cv2.addWeighted(image, 0.5, img, 0.5, 0)

    return unknown2, unknown, grey_matter


