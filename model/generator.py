import tensorflow as tf
from keras.layers import Concatenate, Input, Activation, LeakyReLU, Conv2DTranspose, Concatenate, Dense, Reshape, BatchNormalization, UpSampling2D
from keras.initializers import RandomNormal
from keras.models import Model


def create_generator(noise_dim, embed_dim, channels):
    input_seed = Input(shape=noise_dim, name='input_seed')
    input_embed = Input(shape=embed_dim, name='input_embedding')
    d0 = Dense(128, name='dense_0')(input_embed)
    leaky0 = LeakyReLU(alpha=0.2, name='leaky_relu_0')(d0)

    merge = Concatenate(name='concatenate')([input_seed, leaky0])

    d1 = Dense(256 * 4 * 4, activation="relu", name='dense_1')(merge)
    reshape = Reshape((4, 4, 256), name='reshape')(d1)

    upSamp1 = UpSampling2D(size=(2, 2), name='up_sampling_1')(reshape)
    conv2d1 = Conv2DTranspose(256, kernel_size=5, padding="same", kernel_initializer=RandomNormal(
        stddev=0.02), name='conv2d_transpose_1')(upSamp1)
    batchNorm1 = BatchNormalization(
        momentum=0.8, name='batch_norm_1')(conv2d1)
    leaky1 = LeakyReLU(alpha=0.2, name='leaky_relu_1')(batchNorm1)

    upSamp2 = UpSampling2D(size=(2, 2), name='up_sampling_2')(leaky1)
    conv2d2 = Conv2DTranspose(256, kernel_size=5, padding="same", kernel_initializer=RandomNormal(
        stddev=0.02), name='conv2d_transpose_2')(upSamp2)
    batchNorm2 = BatchNormalization(
        momentum=0.8, name='batch_norm_2')(conv2d2)
    leaky2 = LeakyReLU(alpha=0.2, name='leaky_relu_2')(batchNorm2)

    upSamp3 = UpSampling2D(size=(2, 2), name='up_sampling_3')(leaky2)
    conv2d3 = Conv2DTranspose(128, kernel_size=4, padding="same", kernel_initializer=RandomNormal(
        stddev=0.02), name='conv2d_transpose_3')(upSamp3)
    batchNorm3 = BatchNormalization(
        momentum=0.8, name='batch_norm_3')(conv2d3)
    leaky3 = LeakyReLU(alpha=0.2, name='leaky_relu_3')(batchNorm3)

    upSamp4 = UpSampling2D(size=(4, 4), name='up_sampling_4')(leaky3)
    conv2d4 = Conv2DTranspose(128, kernel_size=4, padding="same", kernel_initializer=RandomNormal(
        stddev=0.02), name='conv2d_transpose_4')(upSamp4)
    batchNorm4 = BatchNormalization(
        momentum=0.8, name='batch_norm_4')(conv2d4)
    leaky4 = LeakyReLU(alpha=0.2, name='leaky_relu_4')(batchNorm4)

    outputConv = Conv2DTranspose(channels, kernel_size=3, padding="same",
                                 kernel_initializer=RandomNormal(stddev=0.02), name='output_conv')(leaky4)
    outputActi = Activation("tanh", name='output_activation')(outputConv)

    model = Model(inputs=[input_seed, input_embed],
                  outputs=outputActi, name='generator_model')

    return model
