import cv2
from numpy import linalg as LA
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
import argparse
import csv

class SegmentObject():

    def __init__(self):
        self.first = True
        self.number = 0

        self.loadData()
        self.procesamiento()

    def loadData(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--dataset", required=True)
        args = vars(ap.parse_args())

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
            else:
                RightX.append(np.sum(img[:,x]))
                
            X.append(x)
            listaX = LeftX + RightX

        return listaX, X, LeftX.index(max(LeftX)), RightX.index(max(RightX))

    def graficY(self,img, h):

        listaY = []
        TopY = []
        BottomY = []
        Y = []

        for y in range(0, h):

            if y < int(h/2):
                TopY.append(np.sum(img[y,:]))
            else:
                BottomY.append(np.sum(img[y,:]))

            Y.append(y)
            listaY = TopY + BottomY

        return listaY, Y, TopY.index(max(TopY)), BottomY.index(max(BottomY))

    def Kernels(self, nmKernel):

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
            # Probar modificar 5x5
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

    def ROI(self, imagen):

        h, w = imagen.shape
        roi_img = np.zeros((h,w))

        for i in range(3, w - 3):
            for j in range(3, h - 3):

                if imagen[j][i]==0:
                    continue 

                select = imagen[j - 3 : j + 4, i - 3 : i + 4] * self.Kernels("delimitar")

                T = sum(sum(select))
                T1 = sum(sum(select[1 : 6, 1 : 6]))
                # 4.5*255 > T
                if ( 4*255 > T or 11*255 < T1 ):
    #            if ( 4.5*255 > T and T>0) or 9*255 < T1 :
                    roi_img[j, i] = 0
         #       elif -7*255 == T  :
          #          roi_img[j, i] = 255
                else:
                    roi_img[j, i] = 255

        return roi_img        

    def histogram(self):

        # Graphics using Pyplot of Matplotlib
        plt.figure(figsize=(2,2))

        plt.subplot(121)
        plt.title("EjeX")
        plt.plot(self.X, self.axisX)

        plt.subplot(122)
        plt.title("EjeY")
        plt.plot(self.Y, self.axisY)

        plt.show()

    def formatCSV(self):
        with open('GenerateData.csv', mode='w') as dataFile:
            self.data = csv.writer(dataFile, delimiter=',', quotechar='\t', quoting = csv.QUOTE_MINIMAL)


    def firstFrame(self, gray, h, w):

        if self.first:

            self.axisX, self.X, x_L, x_R = self.graficX(gray, w)
            self.axisY, self.Y, y_T, y_B = self.graficY(gray, h)

            self.h = h//2
            self.w = w//2
            
            self.y_start = y_T
            self.y_end = y_B + self.h - 20

            self.x_start = x_L
            self.x_end = x_R + self.w - 15
            
            self.first= False


    def saveImg(self, namefile, post_img, cond):

        if cond:
            cv2.imwrite("{}tiff".format(namefile[:-4]), post_img)
        else:
            cv2.imshow("mascara", post_img)
            self.key = cv2.waitKey(0) & 0XFF



    def procesamiento(self):

        for imagePath in self.imagePaths:
            img = cv2.imread(imagePath)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            h,w = np.shape(gray)

            self.firstFrame(gray,h,w)

            gray = gray[self.y_start : self.y_end, self.x_start : self.x_end]

            gray = cv2.GaussianBlur(gray, (5 ,5), 0)

            gray = cv2.filter2D(gray, -1, self.Kernels("filter"))

            _, gray_th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            roi_img = self.ROI(gray_th)

            img_ee = cv2.morphologyEx(roi_img, cv2.MORPH_DILATE, self.Kernels('kernel5'), iterations=2)

            img_ee = np.array(img_ee, np.uint8)

            mask = cv2.bitwise_and(gray, gray, mask = img_ee)

            self.number+=1
            print(self.number)
            self.saveImg(imagePath, mask, False)

    #        if self.key == ord("q"):
     #           break


if __name__ == "__main__":

    objects = SegmentObject()

    cv2.destroyAllWindows()
