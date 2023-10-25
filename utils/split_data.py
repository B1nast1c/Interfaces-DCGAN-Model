import os
import numpy as np
from tensorflow import data
from utils import load_bin, common, process_images

IMG_DIR_PATH = common.IMAGES_LOCATION


def shuffle_data(x_train, y_train):
    train_dataset = data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(
        common.BUFFER_SIZE).batch(common.BATCH_SIZE)
    return train_dataset


def save_dataset(image_dataset, file_name):
    np.save(common.BIN_LOCATION + '/' + file_name,
            image_dataset, allow_pickle=True)


def load_dataset():
    counter = 0
    bin_images = load_bin.load_images()
    bin_labels = load_bin.load_labels()
    train_images_bin, test_images_bin, train_labels_bin, test_labels_bin = [], [], [], []
    indexes = []
    images = os.listdir(IMG_DIR_PATH)

    for single in common.BASE_CLASS:
        indexes.append(len([item for item in images if single in item]))

    for index in indexes:
        test_index = int(index*0.25) - 1

        test_images = bin_images[counter: counter + test_index]
        test_images_bin.extend(test_images)
        train_images = bin_images[counter + test_index: counter + index]
        train_images_bin.extend(train_images)

        test_labels = bin_labels[counter: counter + test_index]
        test_labels_bin.extend(test_labels)
        train_labels = bin_labels[counter + test_index: counter + index]
        train_labels_bin.extend(train_labels)

        counter += index

    test_images_bin = np.array(test_images_bin)
    train_images_bin = np.array(train_images_bin)
    train_images_bin = (train_images_bin.astype(np.float32) - 127.5) / 127.5
    train_images_bin = np.clip(train_images_bin, -1, 1)

    test_labels_bin = np.array(test_labels_bin)
    train_labels_bin = np.array(train_labels_bin)

    save_dataset(test_images_bin, 'images_test.npy')
    save_dataset(train_images_bin, 'images_train.npy')
    save_dataset(train_labels_bin, 'labels_train.npy')
    save_dataset(test_labels_bin, 'labels_test.npy')

    print(train_images_bin.shape)
    print(test_images_bin.shape)
    print(train_labels_bin.shape)
    print(test_labels_bin.shape)

    return (train_images_bin, train_labels_bin), (test_images_bin, test_labels_bin)
