from auxiliar import ImageTreatment
import argparse

import os

def signal_detection():
    files = []

    for file in os.listdir('data'):
        imageTreatment = ImageTreatment.ImageTreatment()
        if not os.path.isfile('data/'+file):
            continue
        files.append('data/' + file)
        args.m = int(args.m) if args.m != '2' else -1
        image = imageTreatment.load_image(direction='data/' + file)
        image = imageTreatment.brute_force(image)
        # hist = imageTreatment.histograma(image=image)
        # #esquinas = imageTreatment.esquinas()
        suavizado = imageTreatment.suavizado(window_size=(int(args.ws), int(args.ws)), sigma_x=float(args.s), image=image)
        # #gradientes = imageTreatment.gradientes(k=int(args.k), image=suavizado) # lista de tres imagens
        #
        edges = imageTreatment.edge_detection(mode='canny', image=suavizado, c_max=float(args.cmax), c_min=float(args.cmin))
        # #closing = imageTreatment.closing(image=edges, k=5)
        # #opening = imageTreatment.opening(image=closing, k=5)


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

