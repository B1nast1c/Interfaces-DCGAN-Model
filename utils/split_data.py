import os
import tensorflow as tf
import numpy as np
from tensorflow import data
from utils import load_bin, common

IMG_DIR_PATH = common.IMAGES_LOCATION
test_size = 0.25


@tf.function
def normalization(tensor):
    tensor = tf.subtract(tf.divide(tensor, 127.5), 1)
    return tensor


def shuffle_data(x_train, y_train):
    train_dataset = data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(
        common.BUFFER_SIZE).batch(common.BATCH_SIZE)
    return train_dataset


def save_dataset(dataset, filename):
    np.save(f'{common.BIN_DATA_LOCATION}{filename}.npy',
            dataset, allow_pickle=True)


def split_dataset():
    bin_images = load_bin.load_data('images')
    bin_labels = load_bin.load_data('labels')
    num_test = int(len(bin_images) * test_size)

    train_images_bin = bin_images[num_test:]
    test_images_bin = bin_images[:num_test]
    train_labels_bin = bin_labels[num_test:]
    test_labels_bin = bin_labels[:num_test]

    test_images_bin = np.array(test_images_bin)
    train_images_bin = np.array(train_images_bin)
    test_labels_bin = np.array(test_labels_bin)
    train_labels_bin = np.array(train_labels_bin)

    save_dataset(test_images_bin, 'images_test.npy')
    save_dataset(train_images_bin, 'images_train.npy')
    save_dataset(train_labels_bin, 'labels_train.npy')
    save_dataset(test_labels_bin, 'labels_test.npy')

    print('Train images shape:', train_images_bin.shape)
    print('Test images shape:', test_images_bin.shape)
    print('Train labels shape:', train_labels_bin.shape)
    print('Test labels shape:', test_labels_bin.shape)

    return (train_images_bin, train_labels_bin), (test_images_bin, test_labels_bin)
