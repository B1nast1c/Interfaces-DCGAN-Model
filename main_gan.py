from utils import process_dataset, split_data

# Procesamiento de datos (V2) - EJECUTAR AL FINAL DEL PROCESAMIENTO
# process_dataset.process_dataset()
# ----------------------------------------------------------------------------

# Division del dataset + shuffle de la training data
(train_images_bin, train_labels_bin), \
    (test_images_bin, test_labels_bin) = split_data.split_dataset()
train_dataset = split_data.shuffle_data(train_images_bin, train_labels_bin)
# ----------------------------------------------------------------------------

# Entrenamiento
# model_training.train(train_dataset, common.EPOCHS)
