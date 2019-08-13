from centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from imutils import paths
import csv

ct = CentroidTracker(maxDisappeared = 1000)
(H, W) = (None, None)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input Images")

args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args['dataset']))

imagePaths.sort()
number= 0

with open('GenerateData.csv', mode = 'w') as dataFile:
#    data = csv.writer(dataFile, delimiter=',', quotechar='\t', quoting = csv.QUOTE_MINIMAL)
    data = csv.writer(dataFile, delimiter=',', quotechar='\t')


    for imagePath in imagePaths:

        frame = cv2.imread(imagePath,0)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        centroids = []

        componentConnect = cv2.connectedComponentsWithStats(frame, connectivity=8)
        _,_,_,centers = componentConnect

        centerList = centers[1:].astype(int).tolist()

        for txt in centerList:
            c_x = txt[0]
            c_y = txt[1]

            centroids.append(txt)

        row = []
        objects = ct.update(centroids)
        
        row.append("{}".format(imagePath.split('/')[1][:8]))

#        print(row)

        for (objectID, centroid) in objects.items():

            text = "{}".format(objectID)

            cv2.putText(frame, text, (centroid[0] - 5, centroid[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1) 
            cv2.circle(frame, (centroid[0], centroid[1]), 2, (255, 0, 0), -1)
            row.append(objectID)
            row.append(centroid)


        data.writerow(row)
        
        #cv2.imshow("Frame", frame)
        #key = cv2.waitKey(0) & 0xFF

        #if key == ord("q"):
         #       break
#        cv2.imwrite("{}tiff".format(imagePath[:-4]), frame)
        print(number)
        number+=1
            
cv2.destroyAllWindows()

