import cv2
from numpy import linalg as LA
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
import argparse
import csv


class Preprocesamiento():

    def __init__(self,manual=False,thresh=63,y_T=0,y_B=0,x_R=0,x_L=0):
        self.y_T = y_T
        self.y_B = y_B
        self.x_R = x_R
        self.x_L = x_L
        self.thresh = thresh
        
        self.manual=manual
        self.dies = 0 
        self.first_img = True
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



    def imprimir(self,img):
        h,w = np.shape(img)

        for i in range(0,h):
            for j in range(0,w):
                print(img[i][j],end=" ")

            print("\n")


    def Preprocesing(self):

        for imagePath in self.imagePaths:
            img = cv2.imread(imagePath)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            gray = cv2.GaussianBlur(gray,(3,3),0)
            h,w = np.shape(gray)
    
            if self.manual == True:

                    firstImg = 0

                    while slef.dies < 1:

                        img = cv2.imread(self.imagePaths[dies],0)

                        gray = cv2.GaussianBlur(gray,(3,3),0)

                        firstImg += gray[y_T:y_B,x_R:x_L]

            #            _,firstImg = cv2.threshold(firstImg,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                        EE1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                        EE2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
                        EE3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

                    #    firstImg = cv2.morphologyEx(firstImg, cv2.MORPH_DILATE, EE1, iterations = 2)  

                        firstImg = np.float32(firstImg)

                        g_mf = firstImg - min(firstImg.flatten())
                        g_sf = self.thresh*(g_mf/max(g_mf.flatten()))

                        firstImg = np.uint8(firstImg)

                        #cv2.imshow("Imadsray", firstImg)
                        #k = cv2.waitKey(0) & 0xFF
                        #if k == ord("q"):
                         #   break
                        
                        #print(dies)
                        self.dies+=1
                    
                    continue

            else:

                if self.first_img == True:

                    firstImg = 0

                    while self.dies < 1:

                        img = cv2.imread(self.imagePaths[self.dies],0)

                        gray = cv2.GaussianBlur(gray,(3,3),0)

                        if self.first_img == True:

                            ejeX, X, x_L, x_R = self.graficX(gray, w)
                            ejeY, Y, y_T, y_B = self.graficY(gray, h)

                     
                            self.first_img = False

                        firstImg += gray[y_T: y_B + h//2 -15, x_L +25: x_R + w//2 ]
            #            _,firstImg = cv2.threshold(firstImg,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                        EE1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                        EE2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
                        EE3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

                    #    firstImg = cv2.morphologyEx(firstImg, cv2.MORPH_DILATE, EE1, iterations = 2)  

                        firstImg = np.float32(firstImg)

                        g_mf = firstImg - min(firstImg.flatten())
                        g_sf = self.thresh*(g_mf/max(g_mf.flatten()))

                        firstImg = np.uint8(firstImg)

                        cv2.imshow("Imadsray", firstImg)
                        k = cv2.waitKey(0) & 0xFF
                        if k == ord("q"):
                            break
                        
                        #print(dies)
                        self.dies+=1
                    
                    continue
            


            if self.manual == True:
                gray_copy = gray[y_T:y_B,x_R:x_L]
                secondgray = gray[y_T:y_B,x_R:x_L]
            else:
                gray_copy = gray[y_T  : y_B + h//2 -15, x_L +25: x_R + w//2 ]
                secondgray = gray[y_T : y_B + h//2 -15, x_L +25: x_R + w//2 ]

            #_,secondgray = cv2.threshold(secondgray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            secondgray = np.float32(secondgray)
            
            funcG = secondgray - firstImg

            g_m = funcG - min(funcG.flatten())
            g_s = self.thresh*(g_m/max(g_m.flatten()))
            g_s = np.uint8(g_s)
           #Rango 40 - 50
        #    n,gray_th = cv2.threshold(g_s,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
         #   print(n)
            ret,gray_th = cv2.threshold(g_s,20,255,cv2.THRESH_BINARY)
            print(ret)

            EE1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            EE2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
            EE3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

            ee61 = cv2.morphologyEx(gray_th, cv2.MORPH_OPEN, EE3, iterations = 1)  
            # Aveces 2
        #    ee62 = cv2.morphologyEx(ee61, cv2.MORPH_ERODE, EE1, iterations = 1)  
            ee64 = cv2.morphologyEx(ee61, cv2.MORPH_DILATE, EE1, iterations = 2)  
        #    ee64 = cv2.morphologyEx(ee63, cv2.MORPH_ERODE, EE3, iterations = 1#)  
            #ee64 = cv2.morphologyEx(ee64, cv2.MORPH_DILATE, EE1, iterations = 2)  

#            cv2.imshow("mag", gray_copy)
 #           cv2.imshow("Imag", secondgray)
            #cv2.imshow("Imagen Gx", funcG)
        #    cv2.imshow("Imadsg", gray_th)
            #cv2.imshow("Imaen G1", ee61)
        #    cv2.imshow("Imaen G2", ee62)
            #cv2.imshow("Imaen G3", ee63)
  #          cv2.imshow("Imaen G4", ee64)

        #    firstFrame = secondgray

            mask = cv2.bitwise_and(gray_copy, gray_copy, mask=ee64)
            cv2.imwrite("{}tiff".format(imagePath[:-4]), mask)
            #cv2.imshow("Mascara", mask)

            print(imagePath[:-4])
            print(ee64.shape)
  #          k = cv2.waitKey(0) & 0xFF
   #         if k == ord("q"):
    #            break
        #    print(imagePath.split())

if __name__=="__main__":
    
    artemia = Preprocesamiento(thresh=32)
    # Probar con thresh = 32 y threshold=20
    cv2.destroyAllWindows()

