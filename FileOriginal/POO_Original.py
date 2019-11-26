import cv2
from numpy import linalg as LA
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
import argparse
import csv

class Preprocesamiento():

    def __init__(self,manual=False,thresh=64, grafica = False):
        self.thresh = thresh
        self.grafica = grafica
        self.manual=manual
        self.inicio()
        self.Preprocesing()

        
    def inicio(self):

        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--dataset", required=True, help="path to input Images")

        args = vars(ap.parse_args())

        print("Loading ...")

        self.imagePaths = list(paths.list_images(args["dataset"]))
        self.imagePaths.sort()


    def graficX(self, img, w):

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

    def graficY(self, img, h):

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
    def Kernels(self,nmKernel):

        if nmKernel == "cross":

            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))

        elif nmKernel == "morphology":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

        elif nmKernel == "kernel5":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

        else:
            print("[INFO] Error: No ingreso el input \"nmKernel\"")

        return kernel


    def histogram(self):

        plt.figure(figsize=(2,2))

        plt.subplot(121)
        plt.title("EjeX")
        plt.plot(self.X, self.ejeX)

        plt.subplot(122)
        plt.title("EjeY")
        plt.plot(self.Y, self.ejeY)

        plt.show()


    def imprimir(self,img):
        h,w = np.shape(img)
        
        for i in range(0,h):
            for j in range(0,w):
                print(img[i][j],end=" ")

            print("\n")

    def firstImage(self):

        gray_f = cv2.imread(self.imagePaths[0],0)
        self.h,self.w = np.shape(gray_f)

        print(self.h,self.w)

        if self.manual == False:

            self.ejeX, self.X, self.x_L, self.x_R = self.graficX(gray_f, self.w)
            self.ejeY, self.Y, self.y_T, self.y_B = self.graficY(gray_f, self.h)
            self.firstImg = self.cutImg(gray_f)

#            self.histogram()

        else:
            r = cv2.selectROI(gray_f)
            self.y_T = int(r[1])
            self.y_B = int(r[1]+r[3])
            self.x_L = int(r[0])
            self.x_R = int(r[0]+r[2])

#            self.firstImg = gray_f[self.y_T: self.y_B + self.h//2, self.x_L: self.x_R + self.w//2]
            self.firstImg = gray_f[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

        self.firstImg = cv2.GaussianBlur(self.firstImg,(7,7),0)
        self.firstImg = np.float32(self.firstImg)

    def Graficar(self,name,img):

        cv2.imshow("{}".format(name),img)
        
    def cutImg(self,img):

        return img[self.y_T: self.y_B, self.x_L: self.x_R]

    def ReducBits(self,img):
        img = np.float32(img)

        funcG = img - self.firstImg
        g_m = funcG - min(funcG.flatten())
        g_s = self.thresh*(g_m/max(g_m.flatten()))
        g_s = np.uint8(g_s)

        return g_s

    def Preprocesing(self):
        self.firstImage()

        for imagePath in self.imagePaths:
            img = cv2.imread(imagePath)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray,(9,9),0)

#            h,w = np.shape(gray)
    
            gray_copy = self.cutImg(gray)
            secondgray = self.cutImg(gray)

            #gray[y_T:y_B,x_R:x_L]

            g_s = self.ReducBits(secondgray)
           #Rango 40 - 50
#            n,nelson = cv2.threshold(g_s,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)


            ret,gray_th = cv2.threshold(g_s,50,255,cv2.THRESH_BINARY)

            EE1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            EE2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
            EE3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            
#            ee60 = cv2.morphologyEx(gray_th, cv2.MORPH_ERODE, EE1, iterations = 1)  
 #           ee60 = cv2.morphologyEx(gray_th, cv2.MORPH_DILATE, EE1, iterations = 1)  
 #No funciona elieminar el objeto  ee61 = cv2.morphologyEx(gray_th, cv2.MORPH_OPEN, EE3, iterations = 1)  
            ee64 = cv2.morphologyEx(gray_th, cv2.MORPH_DILATE, EE1, iterations = 1)
            ee62 = cv2.morphologyEx(ee64, cv2.MORPH_CLOSE, EE1, iterations = 1)  
            ee62 = cv2.morphologyEx(ee62, cv2.MORPH_ERODE, EE2, iterations = 1)  
            ee64 = cv2.morphologyEx(ee62, cv2.MORPH_DILATE, EE1, iterations = 3)

            mask = cv2.bitwise_and(gray_copy, gray_copy, mask=ee64)

            if self.grafica == True:

#                self.Graficar("ee60",ee60)
#                self.Graficar("ee61",ee61)
                self.Graficar("ee62",ee62)
                self.Graficar("threshold",gray_th)
                self.Graficar("Mascara", mask)
                k = cv2.waitKey(0) & 0xFF
                if k == ord("q"):
                    break

            cv2.imwrite("{}tiff".format(imagePath[:-4]), mask)
            print(imagePath)

if __name__=="__main__":
    
    artemia = Preprocesamiento(manual=True, thresh=64, grafica = False)
    # Probar con thresh = 32 y threshold=20
    cv2.destroyAllWindows()

