import os
import numpy as np
from utils import common, load_bin


def assign_labels():
    files_location = os.path.join(common.IMAGES_LOCATION)
    images = os.listdir(files_location)
    y_labels = []

    for image in range(len(images)):
        image_label = images[image].split('_')[1]
        numeric_label = list(common.KEYWORDS.keys()).index(image_label)
        y_labels.append(numeric_label)

    return y_labels


def save_labels():
    y_labels = assign_labels()
    y_labels = np.array(y_labels)

    print(y_labels.shape)

    np.save(common.BIN_LOCATION + '/labels.npy',
            y_labels, allow_pickle=True)


def process_labels():
    save_labels()
