
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
        return hsv

    def brute_force(self, image=None, show=True):
        if image is None:
            image = self.image
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        # plt.plot(hist)
        # plt.show()
        # ret, thresh1 = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        # self.show_image(thresh1)
        res = image
        self.show_image(res) if show else None
        hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
        # hsv[:,:,2] += 10
        # self.show_image(hsv)
        hsv = cv2.resize(hsv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        self.image = cv2.resize(self.image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        lower_red = self.__hsv_scaling(np.array([270, 10, 10]))
        upper_red = self.__hsv_scaling(np.array([360, 100, 100]))
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        # self.show_image(mask_red) if show else None
        # mask_red = cv2.dilate(mask_red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12)), iterations=1)
        # mask_red = cv2.erode(mask_red,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8)), iterations=2)

        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)))
        # self.show_image(mask_red)

        lower_red = self.__hsv_scaling(np.array([0, 10, 10]))
        upper_red = self.__hsv_scaling(np.array([30, 100, 100]))
        mask_red = mask_red + cv2.inRange(hsv, lower_red, upper_red)
        self.show_image(mask_red, name='red')if show else None

        # mask_red = cv2.dilate(mask_red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)), iterations=1)
        # mask_red = cv2.erode(mask_red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)), iterations=2)

        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12)))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)))
        mask_red = cv2.dilate(mask_red, cv2.getStructuringElement(cv2.MORPH_RECT, (14, 14)), iterations=3)
        self.show_image(mask_red, name='red')if show else None
        self.edge_detection(mask_red, c_max=30, c_min=10, draw_contour=False, color='red')

        # hsv = cv2.bitwise_and(hsv, hsv, mask=cv2.bitwise_not(mask_red))
        # self.show_image(hsv)
        lower_yellow = self.__hsv_scaling(np.array([20, 10, 10]))
        upper_yellow = self.__hsv_scaling(np.array([90, 100, 100]))
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12)))
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12)))
        # mask_yellow = cv2.erode(mask_yellow, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        self.show_image(mask_yellow, name='yellow') if show else None
        mask_yellow = cv2.dilate(mask_yellow, cv2.getStructuringElement(cv2.MORPH_RECT, (14, 14)), iterations=4)
        # self.show_image(mask_yellow, name='yellow') if show else None
        # mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        self.show_image(mask_yellow, name='yellow')
        self.edge_detection(mask_yellow, c_max=30, c_min=10, draw_contour=False, color='yellow')
        # # # hsv = cv2.bitwise_and(hsv, hsv, mask=cv2.bitwisenot(mask_yellow))
        lower_blue = self.__hsv_scaling(np.array([215, 10, 10]))
        upper_blue = self.__hsv_scaling(np.array([248, 100, 100]))
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        self.show_image(mask_blue, name='blue') if show else None
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
        # mask_full = mask_red + mask_yellow + mask_blue
        mask_blue = cv2.dilate(mask_blue, cv2.getStructuringElement(cv2.MORPH_RECT, (14, 14)), iterations=3)
        self.show_image(mask_blue, name='blue')if show else None
        self.edge_detection(mask_blue, c_max=30, c_min=10, draw_contour=False, color='blue')

        # mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        # mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20)))
        # mask_full = cv2.dilate(mask_full, cv2.getStructuringElement(cv2.MORPH_RECT,(14,14)), iterations=3)
        # self.show_image(mask_full, name='full') if show else None
        # 
        # gray = cv2.bitwise_and(gray, gray, mask=mask_full)
        # self.show_image(gray)if show else None
        # return mask_full

    def fine_grain2(self, image=None, area=0, color=''):
        if image is None:
            image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.show_image(image)
        image = self.suavizado(image, window_size=(5, 5), sigma_x=4.0)
        output = cv2.Canny(image, 10, 20)
        output = cv2.dilate(output, None, iterations=1)
        output = cv2.erode(output, None, iterations=1)
        self.show_image(output)
        (_, cnts, _) = cv2.findContours(output.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt_image = self.image.copy()
        cv2.drawContours(cnt_image, cnts, -1, (0, 255, 0), 0)
        self.show_image(cnt_image)
        # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = None
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.0025 * peri, True)

            c = c.reshape(-1, 2)
            print(len(approx))
            cv2.drawContours(image, [c], 0, (255, 0, 0), 0)
            self.show_image(image)


    def fine_grain(self, image=None, area=0, color='red'):
        if image is None:
            image = self.image
        # self.show_image(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # h,s,v = cv2.split(hsv)
        if color == 'red':
            lower_red = self.__hsv_scaling(np.array([270, 10, 10]))
            upper_red = self.__hsv_scaling(np.array([360, 100, 100]))
            mask_red = cv2.inRange(hsv, lower_red, upper_red)
            # self.show_image(mask_red)
            lower_red = self.__hsv_scaling(np.array([0, 10, 10]))
            upper_red = self.__hsv_scaling(np.array([30, 100, 100]))
            mask_red = mask_red + cv2.inRange(hsv, lower_red, upper_red)
            # self.show_image(mask_red)
            # self.show_image(mask_red)
            # mask_red = cv2.erode(mask_red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
            mask_red = cv2.dilate(mask_red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
            # self.show_image(mask_red)
            marked = self.edge_detection(mask_red, c_max=2, c_min=1, draw_contour=True, parent_area=area, capprox=[8], ccolor=(0,0,2555), cperi=0.02)
            for i in range(len(marked)):
                print(marked[i])
                m = marked[i]
                if m[2][0] > m[2][1]+(m[2][1]*10/100) or m[2][1] > m[2][0]+(m[2][0]*10/100):
                    marked = marked[:i - 1] + marked[i + 1:]
        elif color == 'yellow':
            lower_yellow = self.__hsv_scaling(np.array([20, 10, 10]))
            upper_yellow = self.__hsv_scaling(np.array([60, 100, 100]))
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_yellow = cv2.dilate(mask_yellow, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=1)
            # self.show_image(mask_yellow)
            # mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            # self.show_image(mask_yellow)
            mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
            # self.show_image(mask_yellow)
            mask_yellow = cv2.dilate(mask_yellow, cv2.getStructuringElement(cv2.MORPH_RECT,(4,4)), iterations=3)
            # self.show_image(mask_yellow)
            marked = self.edge_detection(mask_yellow, c_max=20, c_min=10, draw_contour=True, parent_area=area, capprox=[3], ccolor=(100,255,255), cperi=0.05)
            # marked = marked + self.edge_detection(mask_yellow, c_max=20, c_min=10, draw_contour=True, parent_area=area, capprox=[4], ccolor=(255,0,0), cperi=0.02)
            for i in range(len(marked)):
                print(marked[i])
                m = marked[i]
                if m[2][0] > m[2][1]*1.5 or m[2][1] > m[2][0]*1.5:
                    marked = marked[:i-1]+marked[i+1:]
        elif color == 'blue':
            lower_blue = self.__hsv_scaling(np.array([215, 10, 10]))
            upper_blue = self.__hsv_scaling(np.array([290, 90, 90]))
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            # mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
            # self.show_image(mask_blue)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
            # self.show_image(mask_blue)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
            # mask_blue = cv2.dilate(mask_blue, cv2.getStructuringElement(cv2.MORPH_RECT,(1,4)), iterations=2)
            # self.show_image(mask_blue)
            marked = self.edge_detection(mask_blue, c_max=20, c_min=10, draw_contour=True, parent_area=area, capprox=[4], ccolor=(54,104,255), cperi=0.02)
            for i in range(len(marked)):
                print(marked[i])
                m = marked[i]
                if m[2][0] > m[2][1]*1.5:
                    m[1] = (255,0,0)
                if m[2][0] > m[2][1]*2.5 or m[2][1] > m[2][0]*2.5:
                    marked = marked[:i-1]+marked[i+1:]
        print(len(marked))
        for m in marked:
            cv2.drawContours(image, [m[0]], -1, m[1], 2)
            # self.show_image(image)
        return image

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

    def edge_detection(self, image=None, mode='canny', c_max=200, c_min=100, draw_contour=False, parent_area=0, color='', capprox=[0], cperi=0, ccolor=(255,255,255)):
        # Detección de bordes
        contours = []
        if image is None:
            image = self.image
        if mode == 'canny':
            image = self.suavizado(image, window_size=(5,5), sigma_x=4.0)
            output = cv2.Canny(image, c_min, c_max)
            output = cv2.dilate(output, None, iterations=1)
            # output = cv2.erode(output, None, iterations=1)
            # self.show_image(output)
            (_,cnts, _) = cv2.findContours(output.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt_image = self.image.copy()
            cv2.drawContours(cnt_image, cnts, -1, (0, 255, 0), 0)
            # self.show_image(cnt_image)
            # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
            screenCnt = None
            for c in cnts:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, cperi * peri, True)


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

                if not draw_contour:
                    x, y, w, h = cv2.boundingRect(c)
                    roi = self.image[y:y + h, x:x + w]
                    new_mask = self.fine_grain(roi, w*h/2, color=color)
                    self.image[y:y + h, x:x + w] = new_mask
                    # self.show_image(self.image)
                else:
                    x, y, w, h = cv2.boundingRect(c)
                    area = w*h/2
                    print("parent_area " + str(parent_area))
                    print("area " + str(area))
                    print("approx " + str(len(approx)))
                    if len(approx) in capprox and area > 100:
                        contours.append([box.astype("int"), ccolor, (w,h)])
                # plt.show()
                # self.show_image(self.image)
                # self.show_image(roi)
            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            # if len(approx) == 4:
            #     screenCnt = approx
            #     break
        self.show_image(self.image) if not draw_contour else None
        return output if not draw_contour else contours

    def load_image(self, direction=''):
        # Carga la imagen.
        self.image = cv2.imread(direction)
        return self.image

    def show_image(self, image_show=None, name='image', wait=True):
        # Muestra la imágen.
        if image_show is not None:
            image_show = cv2.resize(image_show,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
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