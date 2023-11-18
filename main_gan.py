"""Script de ejecución principal"""
from utils import process_dataset, split_data, load_bin, common
from model import model_training, model_testing

# Procesamiento de datos (V2) - EJECUTAR AL FINAL DEL PROCESAMIENTO
process_dataset.process_dataset()
# ----------------------------------------------------------------------------

# Division del dataset + shuffle de la training data
split_data.split_dataset()
train_images_bin = load_bin.load_data('images_train')
train_labels_bin = load_bin.load_data('labels_train')

train_dataset = split_data.shuffle_data(train_images_bin, train_labels_bin)

'''plt.figure(figsize=(10, 10))
for images, _ in train_dataset.take(1):
    for i in range(100):
        ax = plt.subplot(10, 10, i + 1)
        plt.imshow(images[i, :, :, 0].numpy().astype("uint8"), cmap='coolwarm')
        plt.axis("off")
plt.show()'''

# ----------------------------------------------------------------------------

# Estructura de las redes
# print(generator.gan_generator().summary())
# print(discriminator.gan_discriminator().summary())

# ----------------------------------------------------------------------------

# Entrenamiento
# model_training.train(train_dataset, common.EPOCHS)


# Testing
model_testing.test()

# ----------------------------------------------------------------------------

# Métricas
# Métricas -> Modelo Personalizado + ENTRENAR
# metrics.plot_training_metrics()
# metrics.deploy_model()

# Métricas PERSONALIZADAS EJECUTADAS INCEPTION SCORE
"""n_split = [i for i in range(20)]
images = load_bin.load_data('images_generated')
is_avg, is_std, gen_scores = metrics.calculate_inception_score(images)
print('Score GEN ', is_avg, is_std)

images2 = load_bin.load_data('images_test')
is_avg, is_std, test_scores = metrics.calculate_inception_score(images2)
print('Score TEST ', is_avg, is_std)

images3 = load_bin.load_data('images_train')
is_avg, is_std, train_scores = metrics.calculate_inception_score(images3)
print('Score TRAIN ', is_avg, is_std)"""
