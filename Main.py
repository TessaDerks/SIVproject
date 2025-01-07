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
    cv2.imwrite('Results/'+method+'/Brain'+nr+'_1.png', grey)
    cv2.imwrite('Results/'+method+'/Brain'+nr+'_2.png', white)
    cv2.imwrite('Results/'+method+'/Brain'+nr+'_3.png', fluid)

#do segmentation with all 3 different methods for all brain scans and save results
def getAllResults():
    for i in range(20,65):
        nr = str(i)
        img = get_image('Dataset/'+nr+'.png')
        filtered_img = Preprocessing.filter_image(img)
        stripped_img = Preprocessing.skull_stripping(filtered_img)
        cv2.imwrite('Dataset/Stripped/Brain'+nr+'_stripped_and_processed.png', stripped_img)

        kmeans_greymatter,kmeans_whitematter,kmeans_fluid = Kmeans.kmeans(stripped_img)
        save_result(kmeans_greymatter,kmeans_whitematter,kmeans_fluid, 'Kmeans',nr)

        watershed_greymatter, watershed_whitematter, watershed_fluid = Watershed.watershed(stripped_img)
        save_result(watershed_greymatter,watershed_whitematter,watershed_fluid, 'Watershed', nr)

        rg_greymatter, rg_whitematter, rg_fluid = RegionGrowing.findRegions(stripped_img)
        save_result(rg_greymatter,rg_whitematter,rg_fluid, 'RegionGrowing', nr)

# showcase result of one image for selected segmentation method
def getImageResult(nr, algorithm):
    plt.figure(figsize=(8, 6))

    stripped_img = get_image('Dataset/Stripped/Brain'+nr+'_stripped_and_processed.png')
    
    subplot_images(stripped_img, 2, 'filtered and stripped image')
    
    if(algorithm == 'kmeans'):
        kmeans_greymatter,kmeans_whitematter,kmeans_fluid = Kmeans.kmeans(stripped_img)
        plot_result(kmeans_greymatter,kmeans_whitematter,kmeans_fluid, 'Kmeans')
    if(algorithm == 'watershed'):
        watershed_greymatter, watershed_whitematter, watershed_fluid = Watershed.watershed(stripped_img)
        plot_result(watershed_greymatter,watershed_whitematter,watershed_fluid, 'Watershed')
    if(algorithm == 'regiongrowing'):
        rg_greymatter, rg_whitematter, rg_fluid = RegionGrowing.findRegions(stripped_img)
        plot_result(rg_greymatter,rg_whitematter,rg_fluid, 'RegionGrowing')
    
    plt.show()

if __name__ == "__main__":
    nr = '56'
    algorithm = 'kmeans'
    getImageResult(nr,algorithm)


