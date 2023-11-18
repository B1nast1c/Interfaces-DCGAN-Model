"""Elementos comunes"""
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
BACKUP_WEIGHTS = './model/backup_weights/'

labels_file_df = pd.read_csv(LABELS_LOCATION)
LABELS_LIST = set(labels_file_df['topic'].values.tolist())
CLASS_MAP = {}
for index in range(len(LABELS_LIST)):
    CLASS_MAP[index] = list(LABELS_LIST)[index]

CHANNELS = 3
LATENT_DIM = 100  # Parte del vector latente + RUIDO
EPOCHS = 500
BATCH_SIZE = 32
BUFFER_SIZE = 30000
DIMENSION = 128
