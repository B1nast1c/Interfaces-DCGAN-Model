from utils import process_dataset

# Procesamiento de datos (V2) - EJECUTAR AL FINAL DEL PROCESAMIENTO
process_dataset.process_dataset()


# IMAGENES
# process_images.process_images()
# CAPTIONS
# process_captions.process_labels()

# Division del dataset
# (train_images_bin, train_labels_bin), \
#   (test_images_bin, test_labels_bin) = split_data.load_dataset()

# Entrenamiento
# train_dataset = split_data.shuffle_data(train_images_bin, train_labels_bin)
# model_training.train(train_dataset, common.EPOCHS)
