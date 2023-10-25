from utils import common
from keras.models import Model
from keras.initializers import RandomNormal
from keras.layers import Input, Embedding, Reshape, Concatenate, ReLU, BatchNormalization, Conv2DTranspose, Dense, ReLU

con_label = Input(shape=(1,))
latent_vector = Input(shape=(100,))


def label_conditioned_generator(n_classes=len(common.BASE_CLASS), embedding_dim=100):
    label_embedding = Embedding(n_classes, embedding_dim)(con_label)
    nodes = 4 * 4
    label_dense = Dense(nodes)(label_embedding)

    label_reshape_layer = Reshape((4, 4, 1))(label_dense)

    return label_reshape_layer


def latent_input(latent_dim=100):
    nodes = 512 * 4 * 4
    latent_dense = Dense(nodes)(latent_vector)
    latent_dense = ReLU()(latent_dense)
    latent_reshape = Reshape((4, 4, 512))(latent_dense)
    return latent_reshape


def gan_generator():
    label_output = label_conditioned_generator()
    latent_vector_output = latent_input()
    merge = Concatenate()([latent_vector_output, label_output])

    x = Conv2DTranspose(64 * 8, kernel_size=4, strides=2, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_1')(merge)
    x = BatchNormalization(
        momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_1')(x)
    x = ReLU(name='relu_1')(x)

    x = Conv2DTranspose(64 * 4, kernel_size=4, strides=2, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_2')(x)
    x = BatchNormalization(
        momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_2')(x)
    x = ReLU(name='relu_2')(x)

    x = Conv2DTranspose(64 * 2, 4, 2, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_3')(x)
    x = BatchNormalization(
        momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='bn_3')(x)
    x = ReLU(name='relu_3')(x)

    x = Conv2DTranspose(64 * 1, 4, 2, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_4')(x)
    x = BatchNormalization(
        momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='bn_4')(x)
    x = ReLU(name='relu_4')(x)

    out_layer = Conv2DTranspose(3, 4, 2, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, activation='tanh', name='conv_transpose_6')(x)

    model = Model(
        [latent_vector,  con_label], out_layer)

    return model
