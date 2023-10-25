from keras.layers import Input, Embedding, Dense, Reshape, ReLU
from utils import common

con_label = Input(shape=(1,))
latent_vector = Input(shape=(common.LATENT_DIM,))


def label_conditioned_generator(n_classes=len(common.BASE_CLASS), embedding_dim=100):
    label_embedding = Embedding(n_classes, embedding_dim)(con_label)
    # Multiplicacion lineal
    nodes = 4 * 4
    label_dense = Dense(nodes)(label_embedding)

    # RESHAPE canal adicional
    label_reshape_layer = Reshape((4, 4, 1))(label_dense)
    return label_reshape_layer


def latent_input(latent_dim=100):
    # INPUT generador
    nodes = 512 * 4 * 4
    latent_dense = Dense(nodes)(latent_vector)
    latent_dense = ReLU()(latent_dense)
    latent_reshape = Reshape((4, 4, 512))(latent_dense)
    return latent_reshape
