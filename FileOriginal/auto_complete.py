import csv
import numpy as np
import argparse
from imutils import paths


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to input images")

args = vars(ap.parse_args())

print("[INFO] Loading ... ")

imagePaths = list(paths.list_files(args["dataset"]))

imagePaths.sort()

File = [i.split('/')[-1] for i in imagePaths]

for f in File:
    filename = f
    
    cadData = []
    
    dStart = {}
    dEnd = {}

    with open('./DataControl/'+filename) as dataFile:
        data = csv.reader(dataFile)
    
        for row in data:
            cadData.append(row)

    for i in cadData[1:]:
        dEnd.setdefault(str(i[1]), float(i[0]))
    
    keys = list(map(str, np.linspace(1, 90, 90, dtype=int)))
    
    dStart = dict.fromkeys(keys, 0)
    
    dJoin = dict(dStart, **dEnd)
    
    with open('./DataControl/procesado/M'+filename, mode='w') as csv_file:
        fieldnames = [filename, 'Tiempo']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
        writer.writeheader()
        for data in dJoin.items():
            writer.writerow({filename: data[0], 'Tiempo': data[1]})
