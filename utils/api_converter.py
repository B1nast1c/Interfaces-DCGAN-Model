"""Conversion para la API"""
import tensorflowjs as tfjs
from keras.models import load_model
from utils import common


def convert_model():
    """
    Convierte el modelo a Tensorflowjs

    Parameters:

    Returns:

    """
    generator = load_model(common.BACKUP_WEIGHTS + 'gen_299.h5')
    discriminator = load_model(common.BACKUP_WEIGHTS + 'disc_299.h5')
    tfjs.converters.save_keras_model(
        generator, common.BACKUP_WEIGHTS + '/generator')
    tfjs.converters.save_keras_model(
        discriminator, common.BACKUP_WEIGHTS + '/discriminator')
