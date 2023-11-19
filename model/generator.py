"""Generador"""
from keras.models import Model
from keras.initializers import RandomNormal
from keras.layers import Input, Embedding, Reshape, Concatenate, \
    LeakyReLU, BatchNormalization, Conv2DTranspose, Dense
from utils import common

con_label = Input(shape=(1,))
latent_vector = Input(shape=(100,))

# Solamente para el embedding
# NOTA NOTA SUPER NOTA -> Mejorar el modelo para la tesis FINAL (OBVI)


def label_conditioned_generator(n_classes=len(common.LABELS_LIST), embedding_dim=100):
    """
    Label Condicional
    """
    label_embedding = Embedding(n_classes, embedding_dim)(con_label)
    # Como es solamente el embedding, se hacen capas de muy baja resolución \
    # pues solamente se actúa con una label
    nodes = 4 * 4
    label_dense = Dense(nodes)(label_embedding)

    label_reshape_layer = Reshape((4, 4, 1))(label_dense)

    return label_reshape_layer


def latent_input():
    """
    Crea la capa de condición para el discriminador.
    Toma la forma de entrada, el número de clases y la dimensión del embedding.
    Devuelve la capa de entrada de etiqueta y la capa de condición.

    def image_disc(in_shape=(common.DIMENSION, common.DIMENSION, 3)):
    """

    nodes = 1024 * 4 * 4
    latent_dense = Dense(nodes)(latent_vector)
    latent_dense = LeakyReLU()(latent_dense)
    latent_reshape = Reshape((4, 4, 1024))(latent_dense)
    return latent_reshape


def gan_generator():
    """
    Crea y compila el modelo del discriminador.
    No recibe argumentos directos.
    Devuelve el modelo del discriminador.
    """

    label_output = label_conditioned_generator()
    latent_vector_output = latent_input()
    # Representación del tamaño  [4, 4, 1024]
    merge = Concatenate()([latent_vector_output, label_output])

    # upsampling la capa de merge a la imagen de salida
    # El upsampling se hace en factor de 2  / DUPLICAR hasta llegar a las dimensiones deseadas
    x = Conv2DTranspose(64 * 16, kernel_size=4,
                        strides=2, padding='same', kernel_initializer=RandomNormal(
                            mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_1')(merge)
    x = BatchNormalization(momentum=0.1, epsilon=0.8,
                           center=1.0, scale=0.02, name='bn_1')(x)
    x = LeakyReLU(name='relu_1')(x)

    x = Conv2DTranspose(64 * 8, kernel_size=4,
                        strides=2, padding='same', kernel_initializer=RandomNormal(
                            mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_2')(x)
    x = BatchNormalization(momentum=0.1, epsilon=0.8,
                           center=1.0, scale=0.02, name='bn_2')(x)
    x = LeakyReLU(name='relu_2')(x)

    x = Conv2DTranspose(64 * 4, kernel_size=4,
                        strides=2, padding='same', kernel_initializer=RandomNormal(
                            mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_3')(x)
    x = BatchNormalization(momentum=0.1, epsilon=0.8,
                           center=1.0, scale=0.02, name='bn_3')(x)
    x = LeakyReLU(name='relu_3')(x)

    x = Conv2DTranspose(64 * 2, kernel_size=4,
                        strides=2, padding='same', kernel_initializer=RandomNormal(
                            mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_4')(x)
    x = BatchNormalization(momentum=0.1, epsilon=0.8,
                           center=1.0, scale=0.02, name='bn_4')(x)
    x = LeakyReLU(name='relu_4')(x)

    x = Conv2DTranspose(64 * 1, kernel_size=4,
                        strides=2, padding='same', kernel_initializer=RandomNormal(
                            mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_5')(x)
    x = BatchNormalization(momentum=0.1, epsilon=0.8,
                           center=1.0, scale=0.02, name='bn_5')(x)
    x = LeakyReLU(name='relu_5')(x)

    # Para imágenes de 128x128
    x = Conv2DTranspose(3, kernel_size=4, strides=1,
                        padding='same', kernel_initializer=RandomNormal(
                            mean=0.0, stddev=0.02), use_bias=False, activation='tanh',
                        name='conv_transpose_final')(x)

    # define model
    model = Model([latent_vector,  con_label], x)
    return model
