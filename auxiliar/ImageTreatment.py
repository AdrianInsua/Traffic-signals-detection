
import numpy as np
import cv2
from imutils import perspective
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
        hsv[1] = (hsv[1]*255)/100
        hsv[2] = (hsv[2]*255)/100
        print(hsv)
        return hsv

    def brute_force(self, image=None, show=False):
        if image is None:
            image = self.image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_soft = self.suavizado(gray,(15,15), 2.0)
        self.show_image(gray_soft)
        hist = cv2.calcHist([gray_soft], [0], None, [256],[0,255])
        min_max = cv2.minMaxLoc(hist)
        print(min_max)
        print((min_max[2][1]+min_max[3][1])/2)
        ret, thresh1 = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        # self.show_image(thresh1)
        res = cv2.bitwise_and(image, image, mask=thresh1)
        self.show_image(res)
        hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
        # lower_yellow = np.array([0, 30, 30])
        # upper_yellow = np.array([60, 255, 255])
        lower_red = self.__hsv_scaling(np.array([290, 10, 10]))
        upper_red = self.__hsv_scaling(np.array([360, 100, 100]))
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        self.show_image(mask_red) if show else None
        mask_red = cv2.dilate(mask_red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12)), iterations=1)
        mask_red = cv2.erode(mask_red,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8)), iterations=2)
        # mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)))
        # mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
        # self.show_image(mask_red)
        lower_red = self.__hsv_scaling(np.array([0, 10, 10]))
        upper_red = self.__hsv_scaling(np.array([30, 100, 100]))
        mask_red = mask_red + cv2.inRange(hsv, lower_red, upper_red)
        self.show_image(mask_red, name='red')if show else None
        mask_red = cv2.dilate(mask_red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)), iterations=1)
        mask_red = cv2.erode(mask_red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)), iterations=2)
        # mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)))
        # mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(60,60)))
        self.show_image(mask_red, name='red')if show else None
        lower_yellow = self.__hsv_scaling(np.array([10, 5, 5]))
        upper_yellow = self.__hsv_scaling(np.array([90, 100, 100]))
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        self.show_image(mask_yellow, name='yellow') if show else None
        mask_yellow = cv2.dilate(mask_yellow, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=1)
        self.show_image(mask_yellow, name='yellow') if show else None
        mask_yellow = cv2.erode(mask_yellow, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        # mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12)))
        # mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        self.show_image(mask_yellow, name='yellow')if show else None
        # self.show_image(mask_yellow + mask_red)
        lower_blue = self.__hsv_scaling(np.array([215, 10, 10]))
        upper_blue = self.__hsv_scaling(np.array([260, 80, 80]))
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        self.show_image(mask_blue, name='blue') if show else None
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        mask_full = mask_red + mask_yellow + mask_blue
        self.show_image(mask_blue, name='blue')if show else None

        mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20)))
        mask_full = cv2.dilate(mask_full, cv2.getStructuringElement(cv2.MORPH_RECT,(14,14)), iterations=3)
        self.show_image(mask_full, name='full')

        gray = cv2.bitwise_and(gray, gray, mask=mask_full)
        self.show_image(gray)if show else None
        return mask_full

    def fine_grain(self, image=None):
        if image is None:
            image = self.image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)

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
        output = cv2.GaussianBlur(image, ksize=window_size, sigmaX=sigma_x, sigmaY=sigma_x)
        return output

    def edge_detection(self, image=None, mode='canny', c_max=200, c_min=100):
        # Detección de bordes
        if image is None:
            image = self.image
        if mode == 'canny':
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            output = cv2.Canny(image, c_min, c_max)
            output = cv2.dilate(output, None, iterations=1)
            output = cv2.erode(output, None, iterations=1)
            self.show_image(output)
            (_,cnts, _) = cv2.findContours(output.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt_image = self.image.copy()
            cv2.drawContours(cnt_image, cnts, -1, (0, 255, 0), 0)
            self.show_image(cnt_image)
            # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
            screenCnt = None
            for c in cnts:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.025 * peri, True)
                print("area "+str(cv2.contourArea(c)))
                print("approx " + str(len(approx)))

                c = c.reshape(-1, 2)
                # cv2.drawContours(self.image, [c], 0, (255,0,0),0)

                # compute the rotated bounding box of the contour
                box = cv2.minAreaRect(c)
                box = cv2.boxPoints(box)
                box = np.array(box, dtype="int")

                # order the points in the contour such that they appear
                # in top-left, top-right, bottom-right, and bottom-left
                # order, then draw the outline of the rotated bounding
                # box
                box = perspective.order_points(box)

                x, y, w, h = cv2.boundingRect(c)
                roi = self.image[y:y + h, x:x + w]
                new_mask = self.fine_grain(roi)
                self.show_image(roi)
                cv2.drawContours(self.image, [box.astype("int")], -1, (0, 255, 0), 2)
                self.show_image(self.image)
                # plt.show()
                # self.show_image(self.image)
                # self.show_image(roi)
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