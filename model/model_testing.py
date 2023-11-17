"""Testing del modelo generativo"""
import numpy as np
import tensorflow as tf
from keras.models import load_model
from utils import common, process_dataset

conditional_gen = load_model(common.BACKUP_WEIGHTS + 'gen_299.h5')


def generate(text_label):
    """
    Generar imagenes por label
    """
    name2idx = common.CLASS_MAP
    label = list(name2idx.keys())[list(name2idx.values()).index(text_label)]
    label = tf.ones(1) * label
    noise = tf.random.normal([1, 100])
    img = np.array(conditional_gen.predict([noise, label]))
    pred = (img[0, :, :, :] + 1) * 127.5
    pred = np.array(pred)
    return pred.astype(np.uint8)


def test():
    """
    Testing del modelo + creacion de archivos
    """
    test_images = []
    values = []

    # Guardado de imagenes de testing en un archiv npy
    for _, value in common.CLASS_MAP.items():
        test_image = generate(value)
        test_images.append(test_image)
        values.append(value)

    test_images = np.array(test_images)
    test_labels = np.array(values)
    process_dataset.save_data(test_images, 'images_generated')
    process_dataset.save_data(test_labels, 'labels_generated')
