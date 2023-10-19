from tensorflow import random
from keras.layers import Concatenate, Input, Conv2D, Dropout, BatchNormalization, LeakyReLU, Dense, Reshape, Flatten
from keras.initializers import RandomNormal
from keras.models import Model


def create_discriminator(image_shape, embed_dim):
    input_shape = Input(shape=image_shape, name="input_image")
    input_embed = Input(shape=embed_dim, name="input_embedding")

    num_filters = [64, 128, 256, 512]
    conv2d1 = Conv2D(32, kernel_size=4, strides=2, input_shape=image_shape,
                     padding="same", kernel_initializer=RandomNormal(stddev=0.02), name="conv2d_1")(input_shape)
    leaky1 = LeakyReLU(alpha=0.2, name="leaky_relu_1")(conv2d1)

    for i, filters in enumerate(num_filters):
        leaky1 = Dropout(0.25, name=f"dropout_{i + 2}")(leaky1)
        leaky1 = Conv2D(filters, kernel_size=4, strides=2, padding="same",
                        kernel_initializer=RandomNormal(stddev=0.02), name=f"conv2d_{i + 2}")(leaky1)
        leaky1 = BatchNormalization(
            momentum=0.8, name=f"batch_norm_{i + 2}")(leaky1)
        leaky1 = LeakyReLU(alpha=0.2, name=f"leaky_relu_{i + 2}")(leaky1)

    dense_embed = Dense(
        128, kernel_initializer=RandomNormal(stddev=0.02), name="dense_embedding")(input_embed)
    leaky_embed = LeakyReLU(alpha=0.2, name="leaky_embedding")(dense_embed)
    reshape_embed = Reshape(
        (4, 4, 8), name="reshape_embedding")(leaky_embed)  # Ajustar a 128x128

    merge_embed = Concatenate(name="concatenate")([leaky1, reshape_embed])

    leaky5 = Dropout(0.25, name="dropout_6")(merge_embed)
    leaky5 = Conv2D(512, kernel_size=4,
                    kernel_initializer=RandomNormal(stddev=0.02), name="conv2d_6")(leaky5)
    leaky5 = BatchNormalization(momentum=0.8, name="batch_norm_6")(leaky5)
    leaky5 = LeakyReLU(alpha=0.2, name="leaky_relu_6")(leaky5)

    leaky5 = Dropout(0.25, name="dropout_7")(leaky5)
    flatten = Flatten(name="flatten")(leaky5)
    output = Dense(1, activation="sigmoid", name="output")(flatten)

    model = Model(inputs=[input_shape, input_embed],
                  outputs=output, name="discriminator_model")

    return model


def create_discriminator(image_shape, embed_dim):
    input_shape = Input(shape=image_shape)
    input_embed = Input(shape=embed_dim)

    num_filters = [64, 128, 256, 512]
    conv2d1 = Conv2D(32, kernel_size=4, strides=2, input_shape=image_shape,
                     padding="same", kernel_initializer=RandomNormal(stddev=0.02))(input_shape)
    leaky1 = LeakyReLU(alpha=0.2)(conv2d1)

    for filters in num_filters:
        leaky1 = Dropout(0.25)(leaky1)
        leaky1 = Conv2D(filters, kernel_size=4, strides=2, padding="same",
                        kernel_initializer=RandomNormal(stddev=0.02))(leaky1)
        leaky1 = BatchNormalization(momentum=0.8)(leaky1)
        leaky1 = LeakyReLU(alpha=0.2)(leaky1)

    dense_embed = Dense(
        128, kernel_initializer=RandomNormal(stddev=0.02))(input_embed)
    leaky_embed = LeakyReLU(alpha=0.2)(dense_embed)
    reshape_embed = Reshape((4, 4, 8))(leaky_embed)  # Ajustar a 128x128

    merge_embed = Concatenate()([leaky1, reshape_embed])

    leaky5 = Dropout(0.25)(merge_embed)
    leaky5 = Conv2D(512, kernel_size=4,
                    kernel_initializer=RandomNormal(stddev=0.02))(leaky5)
    leaky5 = BatchNormalization(momentum=0.8)(leaky5)
    leaky5 = LeakyReLU(alpha=0.2)(leaky5)

    leaky5 = Dropout(0.25)(leaky5)
    flatten = Flatten()(leaky5)
    output = Dense(1, activation="sigmoid")(flatten)

    model = Model(inputs=[input_shape, input_embed], outputs=output)
    return model


def discriminator_loss(cross_entropy, real_image_real_text, fake_image_real_text, real_image_fake_text):
    real_loss = cross_entropy(random.uniform(
        real_image_real_text.shape, 0.8, 1.0), real_image_real_text)
    fake_loss = (cross_entropy(random.uniform(fake_image_real_text.shape, 0.0, 0.2), fake_image_real_text) +
                 cross_entropy(random.uniform(real_image_fake_text.shape, 0.0, 0.2), real_image_fake_text))/2

    total_loss = real_loss + fake_loss
    return total_loss
