"""Carga de archivos binarios"""
import numpy as np
from utils import common


def load_data(filename):
    """
    Carga datos desde un archivo binario y los devuelve.

    Args:
        filename (str): Nombre del archivo binario a cargar.

    Returns:
        np.ndarray: Datos cargados desde el archivo binario.
    """

    bin_data = np.load(f"{common.BIN_DATA_LOCATION}{filename}.npy")
    return bin_data
