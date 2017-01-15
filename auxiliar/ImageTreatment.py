import imutils as imutils
import numpy as np
import cv2
from matplotlib import pyplot as plt

from ShapeDetector import ShapeDetector


class ImageTreatment:
    def __init__(self):
        """

        :param direction: dirección en la que se encuentra la imagen
        """
        self.image = None
        self.modo = 0
        self.images = []

    def __hsv_scaling(self, hsv):
        hsv[0] = (hsv[0]*180)/360
        # hsv[1] = (hsv[1]*100)/255
        # hsv[2] = (hsv[2]*100)/255
        print(hsv)
        return hsv

    def second_way(self, image=None):
        if image is None:
            image = self.image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        # self.show_image(thresh1)
        res = cv2.bitwise_and(image, image, mask=thresh1)
        # self.show_image(res)
        hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
        # lower_yellow = np.array([0, 30, 30])
        # upper_yellow = np.array([60, 255, 255])
        lower_red = self.__hsv_scaling(np.array([290, 10, 10]))
        upper_red = self.__hsv_scaling(np.array([360, 100, 100]))
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        # self.show_image(mask_red)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
        # self.show_image(mask_red)
        lower_red = self.__hsv_scaling(np.array([0, 10, 10]))
        upper_red = self.__hsv_scaling(np.array([30, 150, 150]))
        mask_red = mask_red + cv2.inRange(hsv, lower_red, upper_red)
        self.show_image(mask_red)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1)))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
        self.show_image(mask_red)
        lower_yellow = self.__hsv_scaling(np.array([30, 10, 10]))
        upper_yellow = self.__hsv_scaling(np.array([90, 255, 255]))
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        # self.show_image(mask_yellow)
        # self.show_image(mask_yellow + mask_red)
        lower_blue = self.__hsv_scaling(np.array([215, 25, 25]))
        upper_blue = self.__hsv_scaling(np.array([240, 255, 255]))
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        mask_full = mask_red + mask_yellow + mask_blue
        # self.show_image(mask_blue)
        self.show_image(mask_full)
        mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (10,10)))
        # mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, cv2.getStructuringElem
        # ent(cv2.MORPH_RECT, (105,105)))
        mask_full = cv2.dilate(mask_full, cv2.getStructuringElement(cv2.MORPH_RECT,(14,14)), iterations=3)
        self.show_image(mask_full)
        gray = cv2.bitwise_and(gray, gray, mask=mask_full)
        self.show_image(gray)
        # self.show_image(gray)
        gray = cv2.equalizeHist(gray)
        self.show_image(gray)
        return gray

    def find_circles(self, image=None):
        if image is None:
            image = self.image
        img = cv2.GaussianBlur(image, (5, 5), 0)

        for i in cv2.split(img):
            canny  = cv2.Canny(i, 25, 100, apertureSize=3)
            kernel = np.ones((3,3), np.uint8)
            dilate = cv2.dilate(canny, kernel=kernel)
            self.show_image(image_show=dilate)
            im2, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                approx = cv2.approxPolyDP(c, 0.0001 * cv2.arcLength(c, True), True)
                if len(approx)>9:
                    print(cv2.contourArea(c))
                    if cv2.contourArea(c) > 150:
                        cv2.drawContours(self.image, [c], 0, (0, 255, 255), 0)
            cv2.imshow('im', self.image)
            k = cv2.waitKey(0)

    def histograma(self, image=None):
        # se obtiene el histograma de la imagen
        if image is None:
            image = self.image
        treshold = 40
        cv2.imshow('im', image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        cv2.namedWindow('h', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('h', 300, 300)
        cv2.imshow('h', hue)
        cv2.namedWindow('s', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('s', 300, 300)
        cv2.imshow('s', sat)
        cv2.namedWindow('v', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('v', 300, 300)
        cv2.imshow('v', val)
        k = cv2.waitKey(0)
        if k == 27: # Cierra las ventanas al pulsar escape
            cv2.destroyAllWindows()
        hist=cv2.calcHist([val],[0], None, [255], [0,255])
        min, max, minLoc, maxLoc = cv2.minMaxLoc(hist)
        vMax = maxLoc[1]
        print(vMax)
        mask = val
        print(mask)
        mask[mask >= (vMax - treshold)] = 255
        mask[mask < (vMax - treshold)] = 0
        kernel = np.ones((5,5), np.uint8)
        dilate = cv2.dilate(mask, kernel, iterations=2)
        mask = dilate
        mask[mask == 255] = 20
        mask[mask == 0] = 255
        mask[mask == 20] = 0
        self.show_image(image_show=mask)
        res = cv2.bitwise_and(image, image, mask=mask)
        self.show_image(image_show=res)
        return res

    def gradientes(self, k=5, image=None):
        # Estudio de los gradientes
        if image is None:
            image = self.image
        self.images.append(cv2.Laplacian(image, cv2.CV_64F))
        self.images.append(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=k))
        self.images.append(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=k))
        return self.images[-3:]

    def esquinas(self, image=None):
        if image is None:
            image = self.image
        if self.modo != 0:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray = np.float32(gray)
        else:
            gray = image
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        if self.modo != 0:
            image[dst>0.01*dst.max()] = [0,0,255]
        else:
            image[dst>0.01*dst.max()] = 255
        self.show_image(image_show=image)
        return image

    def suavizado(self, image=None, window_size=(5,5), sigma_x=0):
        # Estudios de suavizado
        if image is None:
            image = self.image
        # ret, thresh1 = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
        # self.show_image(thresh1)
        # res = cv2.bitwise_and(image, image, mask=thresh1)
        # self.show_image(res)
        image = image - cv2.morphologyEx(image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
        self.show_image(image)
        output = cv2.GaussianBlur(image, ksize=window_size, sigmaX=sigma_x, sigmaY=sigma_x)
        return output

    def edge_detection(self, image=None, mode='canny', c_max=200, c_min=100):
        # Detección de bordes
        if image is None:
            image = self.image
        if mode == 'canny':
            output = cv2.Canny(image, c_min, c_max)
            # output = cv2.dilate(output, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)))
            output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)))
            # output = cv2.morphologyEx(output, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
            self.show_image(output)
            (_,cnts, _) = cv2.findContours(output.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
            screenCnt = None
            for c in cnts:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.001 * peri, True)
                # cv2.drawContours(self.image, [c], 0, (255, 255, 255), 0)
                if len(approx) == 3 and cv2.contourArea(c)>100:
                    cv2.drawContours(self.image, [c], 0, (0, 255, 255), 0)
                elif len(approx) >= 4 and len(approx) <= 7 and cv2.contourArea(c)>100:
                    cv2.drawContours(self.image, [c], 0, (0, 255, 0), 0)
                elif len(approx) > 7 and cv2.contourArea(c)>100:
                    cv2.drawContours(self.image, [c], 0, (255,0,0),0)
                    self.show_image(self.image)
                # if our approximated contour has four points, then
                # we can assume that we have found our screen
                # if len(approx) == 4:
                #     screenCnt = approx
                #     break
        self.show_image(self.image)
        return output

    def load_image(self, direction=''):
        # Carga la imagen.
        self.image = cv2.imread(direction)
        return self.image

    def show_image(self, image_show=None, name='image', wait=True):
        # Muestra la imágen.
        if image_show is not None:
            cv2.imshow(name, image_show)
            if wait:
                self.k = cv2.waitKey(0)  # Espera para el cierre de las imagenes
                self.close_image(name, image_show)
            else:
                cv2.waitKey(100)
        for image in self.images:
            if image is None:
                raise Exception("Se debe cargar una imagen")
            cv2.imshow(name, image)
            if wait:
                self.k = cv2.waitKey(0)  # Espera para el cierre de las imagenes
                self.close_image(name, image)
            else:
                cv2.waitKey(100)

    def close_image(self, name='', image=None):
        if self.k == 27: # Cierra las ventanas al pulsar escape
            cv2.destroyAllWindows()
        elif self.k == ord('q'):
            exit()
        elif self.k == ord('s'): # si se pulsa s guarda la imagen
            print("Elige el nombre el archivo:")
            new_name = input()
            print("Se graba: %s.png...." % new_name)
            name = new_name if new_name != '' else name
            cv2.imwrite('data/proc_images/'+name+'.png', image)
            cv2.destroyAllWindows()