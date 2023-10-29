"""Testing del modelo generativo"""
import numpy as np
import tensorflow as tf
from numpy.random import randn
import matplotlib.pyplot as plt
from model import generator, discriminator
from utils import common

conditional_gen = generator.gan_generator()
conditional_disc = discriminator.gan_discriminator()
conditional_gen.load_weights(common.BACKUP_WEIGHTS + 'gen_199.keras')
conditional_disc.load_weights(common.BACKUP_WEIGHTS + 'disc_199.keras')

NROW = 4
NCOL = 5
fig = plt.figure(figsize=(5, 20))

num_classes = len(common.LABELS_LIST)
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

    for label in range(num_classes):
        row = label // NCOL
        col = label % NCOL

        latent_point = generate_latent_points(100,  1)
        # Crear una etiqueta para la clase actual
        labels = tf.constant([label], dtype=tf.int32)

        # Generar una imagen condicional para la clase actual
        generated_image = conditional_gen(
            [latent_point, labels], training=False)

        # Preprocesar la imagen generada
        generated_image = (generated_image[0, :, :, :] + 1) * 127.5
        generated_image = generated_image.numpy().astype(np.uint8)

        # Crear una subtrama para mostrar la imagen
        # Posición en la cuadrícula
        ax = fig.add_subplot(NROW, NCOL, row * NCOL + col + 1)
        ax.imshow(generated_image, cmap='gray')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')

    plt.show()
