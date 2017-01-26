
import numpy as np
import cv2
from imutils import perspective
from matplotlib import colors
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

from ShapeDetector import ShapeDetector


class ImageTreatment:
    def __init__(self):
        """

        :param direction: dirección en la que se encuentra la imagen
        """
        self.image = None
        self.image_test = None
        self.modo = 0
        self.images = []
        self.new_marked = {'red': [], 'yellow':[], 'blue':[], 'light_blue':[]}

    def __hsv_scaling(self, hsv):
        hsv[0] = (hsv[0]*180)/360
        hsv[1] = (hsv[1]*255)/100
        hsv[2] = (hsv[2]*255)/100
        return hsv

    def test(self):
        self.show_image(self.image_test)
        test = []
        predicted = []
        masks = {'red': None, 'yellow': None, 'blue': None, 'light_blue':None}
        regiones = {'red': 0, 'yellow': 0, 'blue': 0, 'light_blue': 0}
        hsv = cv2.cvtColor(self.image_test, cv2.COLOR_BGR2HSV)
        lower_red = self.__hsv_scaling(np.array([0, 10, 10]))
        upper_red = self.__hsv_scaling(np.array([60, 100, 100]))
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        lower_blue = self.__hsv_scaling(np.array([215, 10, 10]))
        upper_blue = self.__hsv_scaling(np.array([248, 100, 100]))
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        lower_blue = self.__hsv_scaling(np.array([100, 10, 10]))
        upper_blue = self.__hsv_scaling(np.array([140, 100, 100]))
        mask_light_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        lower_yellow = self.__hsv_scaling(np.array([20, 10, 10]))
        upper_yellow = self.__hsv_scaling(np.array([90, 100, 100]))
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        self.show_image(mask_red)
        self.show_image(mask_blue)
        self.show_image(mask_yellow)
        self.show_image(mask_light_blue)
        mask_red = self.edge_detection(mask_red, c_max=30, c_min=10, draw_contour=False, color='red', test=True)
        mask_blue = self.edge_detection(mask_blue, c_max=30, c_min=10, draw_contour=False, color='blue', test=True)
        mask_yellow = self.edge_detection(mask_yellow, c_max=30, c_min=10, draw_contour=False, color='blue', test=True)
        mask_light_blue = self.edge_detection(mask_light_blue, c_max=30, c_min=10, draw_contour=False, color='light_blue', test=True)
        masks['red'] = mask_red
        masks['blue'] = mask_blue
        masks['yellow'] = mask_yellow
        masks['light_blue'] = mask_light_blue
        # print(len(mask_red))
        # print(mask_red[1])
        cf = 0
        t_n = []
        matriz = np.zeros((5,5))
        for c in self.new_marked.keys():
            print(c)
            if len(self.new_marked[c]) > 0:
                if c == 'red':
                    t_n.append('Prohibicion')
                if c == 'yellow':
                    t_n.append('Aviso')
                if c == 'blue':
                    t_n.append('Advertencia')
                if c == 'light_blue':
                    t_n.append('Obligacion')
            for n in range(len(self.new_marked[c])):
                rect = []
                x = self.new_marked[c][n]['xy'][0]+self.new_marked[c][n]['nm'][0][3][0]
                y = self.new_marked[c][n]['xy'][1]+self.new_marked[c][n]['nm'][0][3][1]
                for i in range(self.new_marked[c][n]['nm'][0][2][0]):
                    fila = []
                    for j in range(self.new_marked[c][n]['nm'][0][2][1]):
                        fila.append([x+i, y+j])
                    rect.append(fila)
                i = 0
                if n < len(masks[c]):
                    for m in masks[c][n]:
                        for j in range(len(rect)):
                            if [m[0][0],m[0][1]] in rect[j]:
                                # print([m[0][0], m[0][1]])
                                # print(m)
                                i += 1
                    print("test:"+str(len(masks[c][n])))
                    print("pred:"+str(i))
                    if i > (len(masks[c][n])*.7):
                        regiones[c] += 1
                        if c == 'red':
                            matriz[1][1] += 1
                            predicted.append(1)
                            test.append(1)
                        if c == 'yellow':
                            matriz[2][2] += 1
                            predicted.append(2)
                            test.append(2)
                        if c == 'blue':
                            matriz[3][3] += 1
                            predicted.append(3)
                            test.append(3)
                        if c == 'light_blue':
                            matriz[4][4] += 1
                            predicted.append(4)
                            test.append(4)
                    else:
                        if c == 'red':
                            matriz[1][0] += 1
                            predicted.append(0)
                            test.append(1)
                        if c == 'yellow':
                            matriz[2][0] += 1
                            predicted.append(0)
                            test.append(2)
                        if c == 'blue':
                            matriz[3][0] += 1
                            predicted.append(0)
                            test.append(3)
                        if c == 'light_blue':
                            matriz[4][0] += 1
                            predicted.append(0)
                            test.append(4)
                else:
                    t_n.append('Nada')
                    if c == 'red':
                        matriz[0][1] += 1
                        test.append(0)
                        predicted.append(1)
                    if c == 'yellow':
                        matriz[0][2] += 1
                        test.append(0)
                        predicted.append(2)
                    if c == 'blue':
                        matriz[0][3] += 1
                        test.append(0)
                        predicted.append(3)
                    if c == 'light_blue':
                        matriz[0][4] += 1
                        test.append(0)
                        predicted.append(4)
        print(regiones)
        print(matriz)
        target_names = ['Nada','Prohibicion','Peligro','Informacion', 'Obligacion']
        tick_marks = np.arange(len(target_names))
        cm_norm = matriz.astype('float') / matriz.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm_norm, cmap=plt.cm.Blues, norm=colors.PowerNorm(gamma=1. / 2.), interpolation='None')
        plt.colorbar()
        plt.tight_layout()
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        cr = classification_report(test, predicted, target_names=t_n)
        print(cr)


    def brute_force(self, show=True):
        # gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # ret, thresh1 = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        # self.show_image(thresh1)
        # res = cv2.bitwise_and(image, image, mask=thresh1)
        self.show_image(self.image) if show else None

        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # hsv[:,:,2] += 10
        self.show_image(hsv)

        lower_red = self.__hsv_scaling(np.array([270, 10, 10]))
        upper_red = self.__hsv_scaling(np.array([360, 100, 100]))
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        self.show_image(mask_red) if show else None
        # mask_red = cv2.dilate(mask_red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12)), iterations=1)
        # mask_red = cv2.erode(mask_red,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8)), iterations=2)

        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)))
        self.show_image(mask_red) if show else None

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
        self.show_image(mask_yellow, name='yellow') if show else None
        self.edge_detection(mask_yellow, c_max=30, c_min=10, draw_contour=False, color='yellow')
        # # hsv = cv2.bitwise_and(hsv, hsv, mask=cv2.bitwisenot(mask_yellow))
        lower_blue = self.__hsv_scaling(np.array([215, 45, 10]))
        upper_blue = self.__hsv_scaling(np.array([228, 100, 50]))
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        self.show_image(mask_blue, name='blue') if show else None
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        # mask_full = mask_red + mask_yellow + mask_blue
        mask_blue = cv2.dilate(mask_blue, cv2.getStructuringElement(cv2.MORPH_RECT, (14, 14)), iterations=3)
        self.show_image(mask_blue, name='blue') if show else None
        self.edge_detection(mask_blue, c_max=30, c_min=10, draw_contour=False, color='blue')
        lower_blue = self.__hsv_scaling(np.array([220, 55, 10]))
        upper_blue = self.__hsv_scaling(np.array([238, 100, 30]))
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        self.show_image(mask_blue)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        self.show_image(mask_blue)
        # mask_full = mask_red + mask_yellow + mask_blue
        # self.show_image(mask_blue)
        mask_blue = cv2.dilate(mask_blue, cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16)), iterations=2)
        # self.show_image(mask_blue, name='blue')
        self.edge_detection(mask_blue, c_max=30, c_min=10, draw_contour=False, color='light_blue')
        # self.show_image(self.image)
        print(self.new_marked)
        # mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        # mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20)))
        # mask_full = cv2.dilate(mask_full, cv2.getStructuringElement(cv2.MORPH_RECT,(14,14)), iterations=3)
        # self.show_image(mask_full, name='full') if show else None
        # 
        # gray = cv2.bitwise_and(gray, gray, mask=mask_full)
        # self.show_image(gray)if show else None
        # return mask_full


    def fine_grain(self, image=None, area=0, color='red', xy=[]):
        if image is None:
            image = self.image
        # self.show_image(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # h,s,v = cv2.split(hsv)
        new_marked = []
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
            # mask_red = cv2.erode(mask_red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
            mask_red = cv2.dilate(mask_red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
            # self.show_image(mask_red)
            marked = self.edge_detection(mask_red, c_max=2, c_min=1, draw_contour=True, parent_area=area, capprox=[8], ccolor=(0,0,2555), cperi=0.02)
            for i in range(len(marked)):
                if i > len(marked):
                    break
                print(marked[i])
                m = marked[i]
                if m[2][0] <= (m[2][1]+m[2][1]*10/100) and m[2][1] <= (m[2][0]+m[2][0]*10/100):
                    new_marked.append(m)
        elif color == 'yellow':
            lower_yellow = self.__hsv_scaling(np.array([20, 10, 10]))
            upper_yellow = self.__hsv_scaling(np.array([60, 100, 100]))
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            # mask_yellow = cv2.dilate(mask_yellow, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=1)
            # self.show_image(mask_yellow)
            mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            # self.show_image(mask_yellow)
            mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
            # self.show_image(mask_yellow)
            mask_yellow = cv2.dilate(mask_yellow, cv2.getStructuringElement(cv2.MORPH_RECT,(4,4)), iterations=3)
            # self.show_image(mask_yellow)
            marked = self.edge_detection(mask_yellow, c_max=20, c_min=10, draw_contour=True, parent_area=area, capprox=[3], ccolor=(100,255,255), cperi=0.05)
            # marked = marked + self.edge_detection(mask_yellow, c_max=20, c_min=10, draw_contour=True, parent_area=area, capprox=[4], ccolor=(255,0,0), cperi=0.02)
            for i in range(len(marked)):
                if i > len(marked):
                    break
                print(marked[i])
                m = marked[i]
                if m[2][0] <= (m[2][1]+m[2][1]*20/100) and m[2][1] <= (m[2][0]+m[2][0]*20/100):
                    new_marked.append(m)
        elif color == 'blue':
            lower_blue = self.__hsv_scaling(np.array([215, 10, 10]))
            upper_blue = self.__hsv_scaling(np.array([280, 90, 90]))
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            # mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
            # self.show_image(mask_blue)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
            # self.show_image(mask_blue)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
            # mask_blue = cv2.dilate(mask_blue, cv2.getStructuringElement(cv2.MORPH_RECT,(4,4)), iterations=2)
            # self.show_image(mask_blue)
            marked = self.edge_detection(mask_blue, c_max=20, c_min=10, draw_contour=True, parent_area=area, capprox=[4], ccolor=(54,104,255), cperi=0.02)
            for i in range(len(marked)):
                if i > len(marked):
                    break
                print(marked[i])
                m = marked[i]
                if m[2][0] <= m[2][1]*2.5 or m[2][1] <= m[2][0]*2.5:
                    new_marked.append(m)
        elif color == 'light_blue':
            lower_blue = self.__hsv_scaling(np.array([220, 55, 10]))
            upper_blue = self.__hsv_scaling(np.array([238, 100, 30]))
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            # mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
            # self.show_image(mask_blue)
            # mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
            # self.show_image(mask_blue)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
            # mask_blue = cv2.dilate(mask_blue, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)), iterations=2)
            # self.show_image(mask_blue)
            marked = self.edge_detection(mask_blue, c_max=20, c_min=10, draw_contour=True, parent_area=area, capprox=[8], ccolor=(255,240,0), cperi=0.02)
            for i in range(len(marked)):
                if i > len(marked):
                    break
                print(marked[i])
                m = marked[i]
                if m[2][0] <= (m[2][1] + m[2][1] * 10 / 100) and m[2][1] <= (m[2][0] + m[2][0] * 10 / 100):
                    new_marked.append(m)
        if len(new_marked) > 0:
            self.new_marked[color].append({'xy':xy, 'nm':new_marked})
        for m in new_marked:
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

    def edge_detection(self, image=None, mode='canny', c_max=200, c_min=100, draw_contour=False, parent_area=0, color='', capprox=[0], cperi=0, ccolor=(255,255,255), test=False):
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
            self.show_image(cnt_image) if test else None
            if test:
                return cnts
            # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
            # screenCnt = None
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
                    new_mask = self.fine_grain(roi, w*h/2, color=color, xy = [x,y])
                    self.image[y:y + h, x:x + w] = new_mask
                    # self.show_image(self.image)

                else:
                    x, y, w, h = cv2.boundingRect(c)
                    area = w*h/2
                    # print("parent_area " + str(parent_area))
                    # print("area " + str(area))
                    print("approx " + str(len(approx)))
                    if len(approx) in capprox and area > 100:
                        contours.append([box.astype("int"), ccolor, [w,h], [x,y]])
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
        self.image = cv2.resize(self.image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        self.show_image(self.image)

    def load_image_test(self, direction=''):
        self.image_test = cv2.imread(direction)
        self.image_test = cv2.resize(self.image_test, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        self.show_image(self.image_test)


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