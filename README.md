# SIVproject
Project for Signal, Image &amp; Video course at Trento University

Libraries required for running the code:
- Opencv
- numpy
- matplotlib

Code overview:
- Main.py: class to run the code, can use the getImageResult() to obtain result of segmenting one brain scan, by method of choice or use getAllResults() to apply segmentation on all brain scans with all techniques and save it to folder.
- Preprocessing.py: class with preprocessing method and skull stripping
- Kmeans.py
- Watershed.py
- RegionGrowing.py 

Dataset folder has all brain scans, with another folder of all the brain scans preprocessed and skull stripped. Results folder has the segmented images of the brain scans, divided by algorithm.

