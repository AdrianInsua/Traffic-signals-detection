from auxiliar import ImageTreatment
import argparse

import os

def signal_detection():
    for i in range(1,8):
        imageTreatment = ImageTreatment.ImageTreatment()
        args.m = int(args.m) if args.m != '2' else -1
        imageTreatment.load_image(direction='data/' + str(i) + '.jpg')
        imageTreatment.load_image_test(direction='data/' + str(i) + '_test.jpg')
        imageTreatment.brute_force(True)
        imageTreatment.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detección de señales de tráfico en imágenes.",
                                     formatter_class=argparse.RawTextHelpFormatter)

    execute_group = parser.add_argument_group('Opciones de carga de imagen')
    execute_group.add_argument('-m', default=0,
                               help='''Establece el modo de carga de la imagen:
                                0: Escala de grises
                                1: Color sin alpha channel
                                2: Color con alpha channel''')

    option_group = parser.add_argument_group('Opcienes de preprocesado')
    option_group.add_argument('-k', default=5,
                              help='''tamaño de la ventan de sobel''')
    option_group.add_argument('-w', default=False, action='store_true',
                             help='''determina que se detenga el programa hasta que cierres la visualizacion
                             de la imagen.''')
    option_group.add_argument('-s', default=0.5,
                             help='''define el sigma para el suavizado gausiano''')
    option_group.add_argument('-ws', default=5,
                             help='''define el tamaño de la ventana de la mascara gausiana''')
    option_group.add_argument('-cmax', default=30,
                             help='''define el umbral maximo de canny''')
    option_group.add_argument('-cmin', default=5,
                              help='''define el umbral minimo de canny''')

    args = parser.parse_args()

    signal_detection()

