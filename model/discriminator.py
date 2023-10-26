from utils import common
from keras.models import Model
from keras.initializers import RandomNormal
from keras.layers import Input, Embedding, Dense, Reshape, Concatenate, LeakyReLU, Conv2D, BatchNormalization, Dropout, Flatten

con_label = Input(shape=(1,))
inp_img = Input(shape=(common.DIMENSION, common.DIMENSION, 3))


def label_condition_disc(in_shape=(common.DIMENSION, common.DIMENSION, 3), n_classes=len(common.LABELS_LIST), embedding_dim=100):
    con_label = Input(shape=(1,))
    label_embedding = Embedding(n_classes, embedding_dim)(con_label)

    # Escalar la imagen con activación lineal
    nodes = in_shape[0] * in_shape[1] * in_shape[2]
    label_dense = Dense(nodes)(label_embedding)
    label_reshape_layer = Reshape(
        (in_shape[0], in_shape[1], 3))(label_dense)

    return con_label, label_reshape_layer


def image_disc(in_shape=(common.DIMENSION, common.DIMENSION, 3)):
    inp_image = Input(shape=in_shape)
    return inp_image


def gan_discriminator():
    con_label, label_condition_output = label_condition_disc()
    inp_image_output = image_disc()

    # Concatenar label -> channel (common.DIMENSION, common.DIMENSION, 6)
    merge = Concatenate()([inp_image_output, label_condition_output])

    x = Conv2D(64, kernel_size=4, strides=2, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_1')(merge)
    x = LeakyReLU(0.2, name='leaky_relu_1')(x)

    x = Conv2D(64 * 2, kernel_size=4, strides=4, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_2')(x)
    x = BatchNormalization(momentum=0.1,  epsilon=0.8,
                           center=1.0, scale=0.02, name='bn_1')(x)
    x = LeakyReLU(0.2, name='leaky_relu_2')(x)

    x = Conv2D(64 * 4, 4, 2, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_3')(x)
    x = BatchNormalization(momentum=0.1,  epsilon=0.8,
                           center=1.0, scale=0.02, name='bn_2')(x)
    x = LeakyReLU(0.2, name='leaky_relu_3')(x)

    x = Conv2D(64 * 8, 4, 2, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_5')(x)
    x = BatchNormalization(momentum=0.1,  epsilon=0.8,
                           center=1.0, scale=0.02, name='bn_4')(x)
    x = LeakyReLU(0.2, name='leaky_relu_5')(x)

    flattened_out = Flatten()(x)
    dropout = Dropout(0.4)(flattened_out)
    dense_out = Dense(1, activation='sigmoid')(dropout)

    model = Model([inp_image_output, con_label], dense_out)
    return model
