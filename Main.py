import cv2
import numpy as np
import matplotlib.pyplot as plt
import Preprocessing
import Kmeans
import Watershed
import RegionGrowing

def get_image(image_path):
    image = cv2.imread(image_path) 
    return image

def subplot_images(image, nr, title):
    plt.subplot(2, 3, nr)
    plt.imshow(image)
    plt.title(title)

def plot_result(grey,white,fluid, method):
    subplot_images(grey, 4, method+' grey matter image')
    subplot_images(white, 5, method+' white matter image')
    subplot_images(fluid, 6, method+' fluid image')

def save_result(grey,white,fluid, method, nr):
    cv2.imwrite('Results/'+method+'/greymatter/Brain'+nr+'_grey.png', grey)
    cv2.imwrite('Results/'+method+'/whitematter/Brain'+nr+'_white.png', white)
    cv2.imwrite('Results/'+method+'/fluid/Brain'+nr+'_fluid.png', fluid)

if __name__ == "__main__":
    nr = '6'
    '''
    img = get_image('Dataset/Brain'+nr+'.png')
    plt.figure(figsize=(8, 8))
    # resize, denoise and equalize the image
    filtered_img = Preprocessing.filter_image(img)
    subplot_images(filtered_img, 1, 'filtered image')

    # remove the skull from the image and keep the brain
    stripped_img = Preprocessing.skull_stripping(filtered_img)
    '''
    stripped_img = get_image('Dataset/Brain'+nr+'_stripped_and_processed.png')
    subplot_images(stripped_img, 2, 'filtered and stripped image')
    #cv2.imwrite('Dataset/Brain2_stripped_and_processed.png', stripped_img)

    #kmeans segmentation
    #kmeans_greymatter,kmeans_whitematter,kmeans_fluid = Kmeans.kmeans(stripped_img)
    #plot_result(kmeans_greymatter,kmeans_whitematter,kmeans_fluid, 'Kmeans')
    #save_result(kmeans_greymatter,kmeans_whitematter,kmeans_fluid, 'Kmeans',nr)
    
    #watershed segmentation
    watershed_greymatter, watershed_whitematter, watershed_fluid = Watershed.watershed(stripped_img)
    plot_result(watershed_greymatter,watershed_whitematter,watershed_fluid, 'Watershed')
    #save_result(watershed_greymatter,watershed_whitematter,watershed_fluid, 'Watershed', nr)

    # region growing segmentation
    #rg_greymatter, rg_whitematter, rg_fluid = RegionGrowing.findRegions(stripped_img)
    #plot_result(rg_greymatter,rg_whitematter,rg_fluid, 'RegionGrowing')
    #save_result(rg_greymatter,rg_whitematter,rg_fluid, 'RegionGrowing', nr)

    plt.show()
    