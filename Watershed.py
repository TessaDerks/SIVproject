import cv2
import numpy as np
import Preprocessing


def watershed(image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret,threshold = cv2.threshold(grey_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN,kernel, iterations =2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist, 0.15*dist.max(),255,0)  # play around with these boundaries to get different segments
    sure_fg = np.uint8(sure_fg)

    grey = cv2.subtract(sure_bg, sure_fg)

    ret,markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[grey==255]=0

    white = image.copy()
    markers = cv2.watershed(white,markers)
    white[markers>1] = [255, 128, 0]
 
    fluid = np.zeros_like(image)
    fluid[markers==1]= [153, 104, 255]
    fluid = Preprocessing.findBrainContour(image,fluid)

    return grey, white, fluid


