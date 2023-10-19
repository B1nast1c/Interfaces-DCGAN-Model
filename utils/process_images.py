import os
from utils import common, load_bin
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

IMG_DIR_PATH = common.IMAGES_LOCATION
EXTENSION = common.ALLOWED_EXTENSIONS
IMG_DIM = common.GENERATE_SQUARE


def load_and_scale_images():
    images = []
    files = os.listdir(IMG_DIR_PATH)

    for file in files:
        filepath = os.path.join(IMG_DIR_PATH, file)
        if os.path.isfile(filepath) and file.endswith(EXTENSION[0]):
            image = img_to_array(
                load_img(filepath, target_size=(IMG_DIM, IMG_DIM)))
            image = (image.astype(np.float32) / (255 / 2)) - 1
            images.append(image)

    images = np.array(images)
    return images


def save_images():
    training_data = load_and_scale_images()
    np.save(common.BIN_LOCATION + '/images.npy',
            training_data, allow_pickle=True)


def get_test_images():
    counter = 0
    bin_images = load_bin.load_images()
    test_images_bin = []
    indexes = []
    images = os.listdir(IMG_DIR_PATH)

    for single in common.BASE_CLASS:
        indexes.append(len([item for item in images if single in item]))

    for index in indexes:
        q_images = int(index*0.25) - 1
        test_images_bin.extend(bin_images[counter:counter+q_images])
        counter += index

    test_images_bin = np.array(test_images_bin)
    return test_images_bin, indexes
