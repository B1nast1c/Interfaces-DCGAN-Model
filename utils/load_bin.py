import numpy as np
from utils import common


def load_labels():
    bin_captions = np.load(common.BIN_LOCATION + '/labels.npy')
    return bin_captions


def load_images():
    bin_images = np.load(common.BIN_LOCATION + '/images.npy')
    return bin_images
