"""Testing del modelo generativo"""
import numpy as np
import tensorflow as tf
from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib import gridspec
from model import generator, discriminator
from utils import common

conditional_gen = generator.gan_generator()
conditional_disc = discriminator.gan_discriminator()
conditional_gen.load_weights(common.BACKUP_WEIGHTS + 'gen_199.keras')
conditional_disc.load_weights(common.BACKUP_WEIGHTS + 'disc_199.keras')

nrow = len(common.LABELS_LIST)
ncol = len(common.LABELS_LIST)
fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(nrow, ncol, width_ratios=[1] * len(common.LABELS_LIST),
                       wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845)

# Interpolación


def generate_latent_points(latent_dim, n_samples):
    """
    Genera puntos latentes aleatorios.
    Recibe la dimensión latente y la cantidad de muestras.
    Devuelve los puntos latentes generados.
    """

    x_input = randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input


def interpolate_points(p1, p2, n_steps=len(common.LABELS_LIST)):
    """
    Interpola entre dos puntos.
    Recibe dos puntos, con un número opcional de pasos.
    Devuelve los puntos interpolados.
    """

    ratios = np.linspace(0, 1, num=n_steps)
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return np.asarray(vectors)


def test_model():
    """
    Prueba el modelo y muestra las imágenes generadas.
    No recibe argumentos directos.
    No devuelve valores.
    """
    k = 0
    pts = generate_latent_points(100, 2)
    interpolated = interpolate_points(pts[0], pts[1])
    output = None

    for label in range(len(common.LABELS_LIST)):
        labels = tf.ones(len(common.LABELS_LIST)) * label
        predictions = conditional_gen([interpolated, labels], training=False)
        if output is None:
            output = predictions
        else:
            output = np.concatenate((output, predictions))

    for i in range(len(common.LABELS_LIST)):
        for j in range(len(common.LABELS_LIST)):
            pred = (output[k, :, :, :] + 1) * 127.5
            ax = plt.subplot(gs[i, j])
            pred = np.array(pred)
            ax.imshow(pred.astype(np.uint8), cmap='gray')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.axis('off')
            k += 1

    plt.show()
