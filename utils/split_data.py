"""Dvisión de la información"""
import numpy as np
from tensorflow import data
from utils import load_bin, common

IMG_DIR_PATH = common.IMAGES_LOCATION
TEST_SIZE = 0.25


def shuffle_data(x_train, y_train):
    """
    Esta función toma datos de entrenamiento y etiquetas, los baraja y crea un conjunto de datos.

    Parameters:
        x_train (np.ndarray): Datos de entrenamiento.
        y_train (np.ndarray): Etiquetas de entrenamiento.

    Returns:
        tf.data.Dataset: Conjunto de datos barajado.
    """

    train_dataset = data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(
        common.BUFFER_SIZE).batch(common.BATCH_SIZE)
    return train_dataset


def save_dataset(dataset, filename):
    """
    Esta función guarda un conjunto de datos en un archivo binario.

    Parameters:
        dataset (np.ndarray): Conjunto de datos a guardar.
        filename (str): Nombre del archivo binario.

    Returns:
        None
    """

    np.save(f'{common.BIN_DATA_LOCATION}{filename}.npy',
            dataset, allow_pickle=True)


def split_dataset():
    """
    Esta función divide el conjunto de datos en datos de entrenamiento y prueba

    Returns:
        None
    """

    bin_images = load_bin.load_data('images')
    bin_labels = load_bin.load_data('labels')
    num_test = int(len(bin_images) * TEST_SIZE)

    train_images_bin = bin_images[num_test:]
    test_images_bin = bin_images[:num_test]
    train_labels_bin = bin_labels[num_test:]
    test_labels_bin = bin_labels[:num_test]

    test_images_bin = np.array(test_images_bin)
    train_images_bin = np.array(train_images_bin)
    test_labels_bin = np.array(test_labels_bin)
    train_labels_bin = np.array(train_labels_bin)

    train_images_bin = train_images_bin.astype('float32')
    train_images_bin = (train_images_bin - 127.5) / 127.5

    save_dataset(test_images_bin, 'images_test')
    save_dataset(train_images_bin, 'images_train')
    save_dataset(train_labels_bin, 'labels_train')
    save_dataset(test_labels_bin, 'labels_test')

    print('Train images shape:', train_images_bin.shape)
    print('Test images shape:', test_images_bin.shape)
    print('Train labels shape:', train_labels_bin.shape)
    print('Test labels shape:', test_labels_bin.shape)
