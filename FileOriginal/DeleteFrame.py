import argparse
import numpy as np
from imutils import paths
import cv2
import os

ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", required=True, help = "path to input Images")

args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))

imagePaths.sort()

for imagePath in  imagePaths:
    img = cv2.imread(imagePath)
    
    print(imagePath[:-4],np.mean(img.flatten()),np.std(img.flatten()))
    
    if np.mean(img.flatten()) > 0.85  or np.mean(img.flatten()) == 0.0:
        
        os.remove(imagePath)


