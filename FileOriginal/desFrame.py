import os
import time
import argparse
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to input images")

args = vars(ap.parse_args())

print("[INFO] Loading ... ")

imagePaths = list(paths.list_files(args["dataset"]))

imagePaths.sort()

[os.makedirs(i.strip('.h264'),exist_ok=True) for i in imagePaths]
#[print(i.strip('h264')) for i in imagePaths]


for i in imagePaths:
    print(i)

    #shell = "ffmpeg -i {} -ss 1 -vcodec tiff -vf \"select='eq(pict_type, PICT_TYPE_I)'\" -vsync vfr -filter:v fps=fps={}/1 img_%4d.tiff -hide_banner".format(i,30)
#    shell = "ffmpeg -i {} -ss 1 -vcodec tiff -vsync vfr -filter:v fps=fps={}/1 img_%4d.tiff -hide_banner".format(i,30)
    shell = "ffmpeg -i {} -ss 1 -vcodec tiff -vframes 3600 img_%4d.tiff -hide_banner".format(i)

    os.system(shell)
    time.sleep(10)
    
    os.system("mv *.tiff {}".format(i.strip('.h264')))
    #os.system("rm ./{}/*.h264".format(i.strip('h264')))






