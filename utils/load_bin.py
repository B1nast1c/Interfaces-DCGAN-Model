import numpy as np
from utils import common


def load_data(filename):
    bin_data = np.load(f"{common.IMAGES_LOCATION}{filename}.npy")
    return bin_data
