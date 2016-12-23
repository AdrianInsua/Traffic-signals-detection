import numpy as np
import cv2
from matplotlib import pyplot as plt

class ImageTreatment:
    def __init__(self):
        """

        :param direction: dirección en la que se encuentra la imagen
        """
        self.image = None
        self.modo = 0
        self.images = []

    def histograma(self, image=None):
        # se obtiene el histograma de la imagen
        if image is None:
            image = self.image
        if self.modo != 0:
            plt.figure()
            plt.subplot(3, 2, 1)

            color = ("b", "g", "r")
            plt.title("'Flattened' Color Histogram")
            plt.xlabel("Bins")
            plt.ylabel("# of Pixels")

            for i, col in enumerate(color):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                cdf = hist.cumsum()
                cdf_normalized = cdf * hist.max() / cdf.max()
                print(hist)
                print(np.max(hist[1]))
                plt.subplot(3,2,1)
                plt.plot(cdf_normalized, color=col)
                plt.subplot(3,2,2)
                plt.hist(image[i].flatten(), 256, [0,256], color = col)
                plt.subplot(3,2,3)
                plt.plot(hist, color=col)
            plt.xlim([0, 256])
            plt.subplot(3,2,4)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist_hsv = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            plt.imshow(hist_hsv, interpolation='nearest')
            plt.subplot(3, 2,5)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
        else:
            hist = cv2.calcHist([image], [0], None, [256], [0,256])
        return hist

    def gradientes(self, k=5):
        # Estudio de los gradientes
        self.images.append(cv2.Laplacian(self.image, cv2.CV_64F))
        self.images.append(cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=k))
        self.images.append(cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=k))
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
        self.images.append(image)
        return image

    def suavizado(self, image=None, window_size=(5,5), sigma_x=0):
        # Estudios de suavizado
        if image is None:
            image = self.image
        output = cv2.GaussianBlur(image, ksize=window_size, sigmaX=sigma_x)
        self.images.append(output)
        return output

    def edge_detection(self, image=None, mode='canny', c_max=200, c_min=100):
        # Detección de bordes
        if image is None:
            image = self.image
        if mode == 'canny':
            output = cv2.Canny(image, c_min, c_max)
            self.images.append(output)
        return output


    def load_image(self, mode=0, direction=''):
        # Carga la imagen.
        self.modo = mode
        self.image = cv2.imread(direction, mode)
        self.images.append(self.image)
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
        elif self.k == ord('s'): # si se pulsa s guarda la imagen
            print("Elige el nombre el archivo:")
            new_name = input()
            print("Se graba: %s.png...." % new_name)
            name = new_name if new_name != '' else name
            cv2.imwrite('data/proc_images/'+name+'.png', image)
            cv2.destroyAllWindows()