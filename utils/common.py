import pandas as pd

GAN_MODEL_LOCATION = './model/config'
IMAGES_LOCATION = './files/dataset/images/wireframes/'
LABELS_LOCATION = './files/dataset/images/design_topics.csv'
BIN_DATA_LOCATION = './files/dataset/binary/'
# Actualizacion de pesos luego de cada epoch
EPOCHS_LOCATION = './model/weights_train'
# Guardar imagen luego de cada epoch
IMAGE_EPOCHS_LOCATION = './model/epoch_images'
RESULTS_LOCATION = './model/results'  # Resultados del testing

labels_file_df = pd.read_csv(LABELS_LOCATION)
LABELS_LIST = set(labels_file_df['topic'].values.tolist())

GENERATOR_RES_FACTOR = 4  # (1=32, 2=64, 3=96, 4=128 ...)
GENERATE_SQUARE = 32 * GENERATOR_RES_FACTOR
CHANNELS = 3
LATENT_DIM = 100  # Parte del vector latente + RUIDO
EMBED_SIZE = 100
EPOCHS = 200
BATCH_SIZE = 16
BUFFER_SIZE = 10000
