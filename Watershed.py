import cv2
import numpy as np
import matplotlib.pyplot as plt
import Main

def watershed(image):

    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding to create a binary image
    _, binary_image = cv2.threshold(grey_image, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform morphological operations to close gaps in the foreground and background
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations =2)

    # sure background area
    sure_bg = cv2.dilate(binary_image, kernel, iterations=2)
    #sure_bg = cv2.dilate(opening, kernel, iterations =3)

    # Distance transform
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    #dist = cv2.distanceTransform(binary_image,cv2.DIST_L1,5)

    #foreground area
    ret, sure_fg = cv2.threshold(dist, 0.1 * dist.max(), 255, 0)
    #ret, sure_fg = cv2.threshold(dist, 0.2 * dist.max(), 128, 0)
    #ret, sure_fg = cv2.threshold(dist, 15, 255, cv2.THRESH_BINARY)
    #_, distThresh = cv2.threshold(dist,15,250,cv2.THRESH_BINARY)

    sure_fg = np.uint8(sure_fg)

    # unknown area
    unknown = cv2.subtract(sure_bg, sure_fg)
    unknown2 = cv2.subtract(sure_fg, sure_bg)  ##try some other, addition??
    grey_matter = cv2.subtract(sure_bg, unknown2)

    white_matter = sure_bg
    Main.subplot_images(white_matter, 3, 'white matter')

    ret, markers = cv2.connectedComponents(sure_fg) 
    markers = markers +1
    markers[unknown==255] = 0  
  
    markers = cv2.watershed(image,markers)

    print(np.unique(markers))
    img = image.copy()
    img[markers==-1] = [255,0,0]
    img[markers==2]= [0,0,255]
    img[markers==3]= [0,255,0]
    img[markers==4]= [0,255,255]
    img[markers==1]= [255,255,0]

    return img, unknown, grey_matter

def watershed2(image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret,threshold = cv2.threshold(grey_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN,kernel, iterations =2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist, 0.2*dist.max(),255,0)  # play around with these boundaries to get different segments
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    ret,markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255]=0

    img = image.copy()
    markers = cv2.watershed(img,markers)
    img[markers==-1]=[255,0,0]
    img[markers==2]= [0,0,255]
    img[markers==3]= [0,255,0]
    img[markers==4]= [0,255,255]
    img[markers==1]= [255,255,0]

    return img, unknown, sure_fg


