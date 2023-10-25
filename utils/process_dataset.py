import csv
import cv2
import numpy as np
from utils import common

metadata_file = open(common.LABELS_LOCATION, newline='')
metadata_reader = csv.reader(metadata_file)
next(metadata_reader)


def load_dataset():
    # IMAGES -> Nombre de las imágenes, mas no la imagen como tal
    images = common.labels_file_df['screen_id'].values.tolist()
    labels = common.labels_file_df['topic'].values.tolist()
    np_images = []
    np_labels = []

    for image in images:
        try:
            np_image = cv2.imread(f"{common.IMAGES_LOCATION}{image}.png")
            height, width, _ = np_image.shape
            new_width = int((160 / height) * width)
            np_image = cv2.resize(np_image, (new_width, 160),
                                  interpolation=cv2.INTER_LINEAR)
            new_size = (160, 160)

            # Crear una nueva imagen en blanco del tamaño deseado
            resized_image = np.full(
                (new_size[0], new_size[1], 3), (255, 255, 255), dtype=np.uint8)

            # Calcular las coordenadas para pegar la imagen original en el centro
            x_offset = (new_size[1] - new_width) // 2
            y_offset = 0

            # Pegar la imagen original en el centro de la imagen en blanco
            resized_image[y_offset:y_offset + 160,
                          x_offset:x_offset + new_width] = np_image
            np_images.append(resized_image)

        except Exception as e:
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
    np.save(f"{common.IMAGES_LOCATION}{filename}.npy",
            data, allow_pickle=True)


def process_dataset():
    load_dataset()
