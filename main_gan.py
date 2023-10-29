"""Script de ejecución principal"""
import matplotlib.pyplot as plt
from utils import process_dataset, split_data, load_bin, common
from model import generator, discriminator, model_training, model_testing

'''
# Procesamiento de datos (V2) - EJECUTAR AL FINAL DEL PROCESAMIENTO
process_dataset.process_dataset()
# ----------------------------------------------------------------------------

# Division del dataset + shuffle de la training data
split_data.split_dataset()
train_images_bin = load_bin.load_data('images_train')
train_labels_bin = load_bin.load_data('labels_train')

train_dataset = split_data.shuffle_data(train_images_bin, train_labels_bin)

plt.figure(figsize=(10, 10))
for images, _ in train_dataset.take(1):
    for i in range(100):
        ax = plt.subplot(10, 10, i + 1)
        plt.imshow(images[i, :, :, 0].numpy().astype("uint8"), cmap='coolwarm')
        plt.axis("off")
plt.show()

# ----------------------------------------------------------------------------

# Estructura de las redes
print(generator.gan_generator().summary())
print(discriminator.gan_discriminator().summary())

# ----------------------------------------------------------------------------

# Entrenamiento
model_training.print_inputs_outputs()
model_training.train(train_dataset, common.EPOCHS)
'''

# Testing
model_testing.test_model()
