"""Archivo de procesamiento de imagenes + labels"""
import csv
import cv2
import numpy as np
from utils import common

metadata_file = open(common.LABELS_LOCATION, newline='', encoding='utf-8')
metadata_reader = csv.reader(metadata_file)
next(metadata_reader)


def load_dataset():
    """
    Carga imágenes, las redimensiona y almacena en formato binario.
    No recibe argumentos directos.
    Almacena imágenes y etiquetas en archivos binarios.
    Maneja excepciones cv2.error para errores de imagen.

    """

    # IMAGES -> Nombre de las imágenes, mas no la imagen como tal
    images = common.labels_file_df['screen_id'].values.tolist()
    labels = common.labels_file_df['topic'].values.tolist()
    np_images = []
    np_labels = []

    for image in images:
        try:
            np_image = cv2.imread(f"{common.IMAGES_LOCATION}{image}.png")
            height, width, _ = np_image.shape
            new_width = int((common.DIMENSION / height) * width)
            np_image = cv2.resize(np_image, (new_width, common.DIMENSION),
                                  interpolation=cv2.INTER_LINEAR)
            new_size = (common.DIMENSION, common.DIMENSION)

            # Crear una nueva imagen en blanco del tamaño deseado
            resized_image = np.full(
                (new_size[0], new_size[1], 3), (255, 255, 255), dtype=np.uint8)

            # Calcular las coordenadas para pegar la imagen original en el centro
            x_offset = (new_size[1] - new_width) // 2
            y_offset = 0

            # Pegar la imagen original en el centro de la imagen en blanco
            resized_image[y_offset:y_offset + common.DIMENSION,
                          x_offset:x_offset + new_width] = np_image
            np_images.append(resized_image)

        except FileNotFoundError as e:
            print(f"Error procesando la imagen {image}: {str(e)}")

    np_images = np.array(np_images)

    for label in labels:
        if label in common.LABELS_LIST:
            label_index = list(common.LABELS_LIST).index(label)
            np_labels.append(label_index)

    np_labels = np.array(np_labels)
    print(np_images.shape, np_labels.shape)
    save_data(np_images, 'images')
    save_data(np_labels, 'labels')


def save_data(data, filename):
    """
    Guarda datos en formato binario en un archivo.
    Toma un conjunto de datos y un nombre de archivo.
    Útil para almacenar imágenes y etiquetas en archivos binarios.
    """

    np.save(f"{common.BIN_DATA_LOCATION}{filename}.npy",
            data, allow_pickle=True)


def process_dataset():
    """
    Guarda los datos en formato binario en un archivo.
    Args:
        data: Los datos que se desean guardar en formato binario.
        filename: El nombre del archivo en el que se guardarán los datos.
    Returns:
        No retorna ningún valor. Guarda los datos en el archivo especificado.
    """
    load_dataset()
