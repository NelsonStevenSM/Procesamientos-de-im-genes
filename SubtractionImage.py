import cv2
from numpy import linalg as LA
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
import argparse
import csv

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

    elif nmKernel == "cross":

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))

    elif nmKernel == "filter":
        kernel = np.array([
            [0,1,0],
            [1,1,1],
            [0,1,0]],dtype=np.uint8)/5

    elif nmKernel == "morphology":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

    elif nmKernel == "kernel5":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    else:
        print("[INFO] Error: No ingreso el input \"nmKernel\"")

    return kernel

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

number = 0


def imprimir(img):

    h,w = np.shape(img)

    for i in range(0,h):
        for j in range(0,w):
            print(img[i][j],end=" ")

        print("\n")


dies = 0 
first = True

for imagePath in imagePaths:
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    fix = max(np.std(gray, axis=0))
    fiy = min(np.std(gray, axis=1))
    gray = cv2.GaussianBlur(gray,(3,3),sigmaX = fix, sigmaY = fiy)
    #gray = cv2.GaussianBlur(gray,(3,3),0)

    h,w = np.shape(gray)

    if first == True:
        firstImg = 0
        #while dies < 100:
        while dies < 2:

            img = cv2.imread(imagePaths[dies],0)
            fix = max(np.std(gray, axis=0))
            fiy = min(np.std(gray, axis=1))

            #gray = cv2.GaussianBlur(gray,(3,3),0)

            gray = cv2.GaussianBlur(gray,(3,3),sigmaX = fix, sigmaY = fiy)

            if first == True:

                ejeX, X, x_L, x_R = graficX(gray, w)
                ejeY, Y, y_T, y_B = graficY(gray, h)

                histogram(ejeX,X, ejeY, Y)
                first= False

            firstImg += gray[y_T: y_B + h//2 + 15, x_L + 5: x_R + w//2 - 5]
            _,firstImg = cv2.threshold(firstImg,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            EE1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            EE2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
            EE3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

            firstImg = cv2.morphologyEx(firstImg, cv2.MORPH_DILATE, EE1, iterations = 2)  

            firstImg = np.float32(firstImg)

            g_mf = firstImg - min(firstImg.flatten())
            g_sf = 63*(g_mf/max(g_mf.flatten()))

            firstImg = np.uint8(firstImg)

            cv2.imshow("Imadsray", firstImg)
            k = cv2.waitKey(0) & 0xFF
            if k == ord("q"):
                break
            
            print(dies)
            dies+=1
        
        continue
    


#    gray = cv2.GaussianBlur(gray,(3,3),0)
    
    gray_copy = gray[y_T  : y_B + h//2 + 15 , x_L + 5: x_R + w//2 - 5]
    secondgray = gray[y_T  : y_B + h//2 + 15 , x_L + 5: x_R + w//2 - 5]
    _,secondgray = cv2.threshold(secondgray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    secondgray = np.float32(secondgray)
    
    funcG = secondgray - firstImg

    g_m = funcG - min(funcG.flatten())
    g_s = 63*(g_m/max(g_m.flatten()))
    g_s = np.uint8(g_s)
   #Rango 40 - 50 
    ret,gray_th = cv2.threshold(g_s,45,255,cv2.THRESH_BINARY)

    EE1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    EE2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    EE3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    ee6 = cv2.morphologyEx(gray_th, cv2.MORPH_OPEN, EE3, iterations = 1)  
    # Aveces 2
    ee6 = cv2.morphologyEx(ee6, cv2.MORPH_ERODE, EE1, iterations = 1)  
    ee6 = cv2.morphologyEx(ee6, cv2.MORPH_DILATE, EE1, iterations = 2)  

    cv2.imshow("Img1 Gray", gray_copy)
    cv2.imshow("Img2 Second", secondgray)
    cv2.imshow("Imagen Gx", funcG)
    cv2.imshow("Imadsg", gray_th)
    cv2.imshow("Imaen Gx", ee6)

#    firstFrame = secondgray
#    cv2.imwrite("{}tiff".format(imagePath[:-4]), ee4)

    mask = cv2.bitwise_and(gray_copy,gray_copy,mask=ee6)
    cv2.imshow("Mascara", mask)

    print(imagePath[:-4])
    print(ee6.shape)
    k = cv2.waitKey(0) & 0xFF
    if k == ord("q"):
        break
#    print(imagePath.split())
cv2.destroyAllWindows()

