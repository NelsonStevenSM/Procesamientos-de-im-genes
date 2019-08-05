import cv2
from numpy import linalg as LA
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input Images")

args = vars(ap.parse_args())

print("Loading ...")

imagePaths = list(paths.list_images(args["dataset"]))
imagePaths.sort()

# Height and Weight

def graficX(img, w):

    listaX = []
    LeftX = []
    RightX = []
    X = []

    for x in range(0, w):
        if x < int(w/2):
            LeftX.append(np.sum(img[:,x]))
            X.append(x)
        else:
            RightX.append(np.sum(img[:,x]))
            X.append(x)

        listaX = LeftX + RightX

    return listaX, X, LeftX.index(max(LeftX)), RightX.index(max(RightX))

def graficY(img, h):

    listaY = []
    TopY = []
    BottomY = []
    Y = []

    for y in range(0, h):

        if y < int(h/2):
            TopY.append(np.sum(img[y,:]))
            Y.append(y)
        else:
            BottomY.append(np.sum(img[y,:]))
            Y.append(y)

        listaY = TopY + BottomY

    return listaY, Y, TopY.index(max(TopY)), BottomY.index(max(BottomY))

# Section of Kernels
def Kernels(nmKernel):

    if nmKernel == "delimitar":
        kernel = np.array([
            [-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,1,1,1,-1,-1],
            [-1,1,1,1,1,1,-1],
            [-1,1,1,1,1,1,-1],
            [-1,1,1,1,1,1,-1],
            [-1,-1,1,1,1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1]])

    elif nmKernel == "filter":
        kernel = np.array([
            [0,1,0],
            [1,1,1],
            [0,1,0]],dtype=np.uint8)/5

    elif nmKernel == "morphology":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))

    else:
        print("[INFO] Error: No ingreso el input \"nmKernel\"")

    return kernel

# Delimitar la región de Interés
def ROI(imagen):

    h, w = imagen.shape
    roi_img = np.zeros((h, w))

    for i in range(3, w - 3):
        for j in range(3, h - 3):

            if imagen[j][i]==0:
                continue 

            select = imagen[j - 3 : j + 4, i - 3 : i + 4] * Kernels("delimitar")

            T = sum(sum(select))
            T1 = sum(sum(select[1 : 6, 1 : 6])) + select[1, 1] + select[1, 5] + select[5, 1] + select[5, 5]

            # Segmentando la región de interés usando umbrales T, T1

            if ( 4.5*255 > T or 9*255 < T1 ):
                roi_img[j, i] = 0
            elif -7*255 == T  :
                roi_img[j, i] = 255
            else:
                roi_img[j, i] = 255

    return roi_img        

def histogram(axisX, X, axisY, Y):

    # Graphics using Pyplot of Matplotlib
    plt.figure(figsize=(2,2))

    plt.subplot(121)
    plt.title("EjeX")
    plt.plot(X, axisX)

    plt.subplot(122)
    plt.title("EjeY")
    plt.plot(Y, axisY)

    plt.show()

# Creamos una Matriz 
A = np.zeros((354, 137*1437))
rng_current = 0
rng_update =137
first = True

for imagePath in imagePaths:
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h,w = np.shape(gray)
    if first == True:
        ejeX, X, x_L, x_R =graficX(gray, w)
        ejeY, Y, y_T, y_B = graficY(gray, h)

        first= False
    
    print("{}, H : {}, W : {}".format(imagePath,((h//2 - 10) - y_B + 1),(x_R + w//2 - x_L + 1)))


    # Determination for selection ROI

    gray = gray[y_T: y_B + h//2 - 20, x_L : x_R + w//2]

    gray = cv2.filter2D(gray, -1, Kernels("filter"))

    _,gray_th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    roi_img = ROI(gray_th)

    roi_img = cv2.morphologyEx(roi_img, cv2.MORPH_DILATE, Kernels('morphology'), iterations=2)

    roi_img = np.array(roi_img, np.uint8)

    msk = cv2.bitwise_and(gray, gray, mask = roi_img)

    # Show of Image

    #cv2.imshow("Imagen Gray", gray)
    #cv2.imshow("Gray-Threshold",gray_th)
    #cv2.imshow("ROI",roi_img)
    #cv2.imshow("msk",msk)

    fig = plt.figure()

    a = fig.add_subplot(2, 2, 1)
    imgplot = plt.imshow(gray, cmap='hot')
    a.set_title('Imagen Gray - Original')

    a = fig.add_subplot(2, 2, 2)
    imgplot = plt.imshow(gray_th,cmap='hot')
    a.set_title('Gray - Threshold')


    a = fig.add_subplot(2, 2, 3)
    imgplot = plt.imshow(roi_img, cmap='hot')
    a.set_title('ROI')

    a = fig.add_subplot(2, 2, 4)
    imgplot = plt.imshow(msk)
    #imgplot.set_clim(0.0, 0.7)
    a.set_title('Mascara')

    plt.show()

    A[:,rng_current:rng_update] = msk
    rng_current = rng_update
    rng_update += 137
    

cv2.imwrite("./img_Proc.png",A)
cv2.waitKey(0) & 0XFF
cv2.destroyAllWindows()

