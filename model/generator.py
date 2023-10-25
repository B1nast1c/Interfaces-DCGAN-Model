from model import input_processing
from keras.models import Model
from keras.initializers import RandomNormal
from keras.layers import Concatenate, LeakyReLU, BatchNormalization, Conv2DTranspose


def gan_generator():
    label_output = input_processing.label_conditioned_generator()
    latent_vector_output = input_processing.latent_input()
    merge = Concatenate()([latent_vector_output, label_output])

    x = Conv2DTranspose(64 * 8, kernel_size=4, strides=2, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_1')(merge)
    x = BatchNormalization(
        momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_1')(x)
    x = LeakyReLU(name='relu_1')(x)

    x = Conv2DTranspose(64 * 4, kernel_size=4, strides=2, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_2')(x)
    x = BatchNormalization(
        momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_2')(x)
    x = LeakyReLU(name='relu_2')(x)

    x = Conv2DTranspose(64 * 2, 4, 2, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_3')(x)
    x = BatchNormalization(
        momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='bn_3')(x)
    x = LeakyReLU(name='relu_3')(x)

    x = Conv2DTranspose(64 * 1, 4, 2, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_4')(x)
    x = BatchNormalization(
        momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='bn_4')(x)
    x = LeakyReLU(name='relu_4')(x)

    out_layer = Conv2DTranspose(3, 4, 2, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, activation='tanh', name='conv_transpose_6')(x)

    model = Model(
        [input_processing.latent_vector,  input_processing.con_label], out_layer)

    return model
