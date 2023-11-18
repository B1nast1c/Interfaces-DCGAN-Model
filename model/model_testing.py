"""Testing del modelo generativo"""
import numpy as np
from numpy.random import randn, randint
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
from utils import common, process_dataset

conditional_gen = load_model(
    common.BACKUP_WEIGHTS + 'gen_499.h5', compile=False)


def generate_latent_points(latent_dim, n_samples, n_classes=20):
    """
    Puntos latentes de la imagen
    """
    x_input = randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


def generate(text_label):
    """
    Generar imagenes por label
    """
    name2idx = common.CLASS_MAP
    label = list(name2idx.keys())[list(name2idx.values()).index(text_label)]
    num_label = tf.ones(1) * label
    noise = tf.random.normal([1, 100])
    img = np.array(conditional_gen.predict([noise, num_label]))
    pred = (img[0, :, :, :] + 1) * 127.5
    pred = np.array(pred)
    return pred.astype(np.uint8), label


def test():
    """
    Testing del modelo + creacion de archivos
    """
    test_images = []
    values = []
    _, axs = plt.subplots(5, 4, figsize=(12, 15))
    axs = axs.flatten()

    # Guardado de imagenes de testing en un archiv npy
    for i, value in common.CLASS_MAP.items():
        test_image, num_val = generate(value)
        test_images.append(test_image)
        values.append(num_val)
        axs[i].imshow(test_image, cmap='gray')
        axs[i].axis('off')

    plt.show()

    test_images = np.array(test_images)
    test_labels = np.array(values)
    process_dataset.save_data(test_images, 'images_generated')
    process_dataset.save_data(test_labels, 'labels_generated')
