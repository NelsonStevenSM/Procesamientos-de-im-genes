from centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import os
import cv2
from imutils import paths
import csv

ct = CentroidTracker(maxDisappeared = 500)
(H, W) = (None, None)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input Images")

args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args['dataset']))

imagePaths.sort()
number= 0
count_frame=0

#filename = "{}.csv".format(imagePath.split('/')[2])
#filename = "{}.csv".format(args['dataset'].strip(".tiff").split('/')[-1])
filename = "{}.csv".format("N"+args['dataset'].split('/')[-2])
#if False:
with open(filename, mode = 'w') as dataFile:
    data = csv.writer(dataFile, delimiter=',', quotechar='\t')

    for imagePath in imagePaths:
        count_frame+=1
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
        
        row.append(count_frame)
        row.append("{}".format(imagePath.strip(".tiff").split('/')[-1]))

        print(imagePath.strip(".tiff").split('/')[-1])
        #row.append(len(objects))
        for (objectID, centroid) in objects.items():
    #            row.append(objectID is "")
            
            text = "{}".format(objectID)

            cv2.putText(frame, text, (centroid[0] - 5, centroid[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1) 
            cv2.circle(frame, (centroid[0], centroid[1]), 2, (255, 0, 0), -1)
            row.append(objectID)
            row.append(centroid[0])
            row.append(centroid[1])
        
        data.writerow(row)
        
        #cv2.imshow("Frame", frame)
        #key = cv2.waitKey(0) & 0xFF

        #if key == ord("q"):
         #       break
#        cv2.imwrite("{}tiff".format(imagePath[:-4]), frame)
        print(number)
        number+=1


    dataFile.close()

print(filename)

header = list(objects.keys())

string = 'frame,' + 'imagen'
for i in header:

    string += ",id{0},x{0},y{0}".format(i)


command = "echo | awk \'BEGIN {0}print \"{1}\"{2} {0}print $0{2}\' {3} >> Mod_{3}".format("{",string,"}", filename)

print(command)

os.system(command)

command = "rm {}".format(filename)

os.system(command)


cv2.destroyAllWindows()

