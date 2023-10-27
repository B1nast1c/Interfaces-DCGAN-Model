"""Entrenamiento del modelo generativo"""
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from model import generator, discriminator
from utils import common

# PARAMETROS
# ----------------------------------------------------------------

conditional_gen = generator.gan_generator()
conditional_disc = discriminator.gan_discriminator()
NUM_EXAMPLES_TO_GENERATE = 5
generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, common.LATENT_DIM])
binary_cross_entropy = BinaryCrossentropy()


def print_inputs_outputs():
    '''print(conditional_gen.input)
    print(conditional_disc.input)
    print(conditional_gen.output)
    print(conditional_disc.output)'''

    print(seed.shape)


def generator_loss(label, fake_output):
    """
    Calcula la pérdida del generador.

    Parameters:
        label: Las etiquetas reales.
        fake_output: Salida falsa del discriminador.

    Returns:
        gen_loss: La pérdida del generador.
    """

    gen_loss = binary_cross_entropy(label, fake_output)
    return gen_loss


def discriminator_loss(label, output):
    """
    Calcula la pérdida del discriminador.

    Parameters:
        label: Las etiquetas reales o falsas.
        output: Salida del discriminador.

    Returns:
        disc_loss: La pérdida del discriminador.
    """

    disc_loss = binary_cross_entropy(label, output)
    return disc_loss

# ----------------------------------------------------------------


@tf.function
def train_step(images, target):
    """
    Realiza un paso de entrenamiento.

    Parameters:
        images: Imágenes reales.
        target: Objetivos o etiquetas para las imágenes.

    Returns:
        None
    """

    noise = tf.random.normal([target.shape[0], common.LATENT_DIM])

    with tf.GradientTape() as disc_tape1:
        generated_images = conditional_gen([noise, target], training=True)
        real_output = conditional_disc([images, target], training=True)
        real_targets = tf.ones_like(real_output)
        disc_loss1 = discriminator_loss(real_targets, real_output)

    # Calculo de las gradientes para las etiquetas reales
    gradients_of_disc1 = disc_tape1.gradient(
        disc_loss1, conditional_disc.trainable_variables)

    # parameters optimization for discriminator for real labels
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_disc1, conditional_disc.trainable_variables))

    # Entrenar discriminador con etiquetas erróneas
    with tf.GradientTape() as disc_tape2:
        fake_output = conditional_disc(
            [generated_images, target], training=True)
        fake_targets = tf.zeros_like(fake_output)
        disc_loss2 = discriminator_loss(fake_targets, fake_output)

    gradients_of_disc2 = disc_tape2.gradient(
        disc_loss2, conditional_disc.trainable_variables)

    discriminator_optimizer.apply_gradients(zip(gradients_of_disc2,
                                                conditional_disc.trainable_variables))

    # Entrenar discriminador con etiquetas reales
    with tf.GradientTape() as gen_tape:
        generated_images = conditional_gen([noise, target], training=True)
        fake_output = conditional_disc(
            [generated_images, target], training=True)
        real_targets = tf.ones_like(fake_output)
        gen_loss = generator_loss(real_targets, fake_output)

    gradients_of_gen = gen_tape.gradient(
        gen_loss, conditional_gen.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_gen,
                                            conditional_gen.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
    """
    Realiza el entrenamiento del modelo generativo condicional.

    Parameters:
        dataset: Conjunto de datos de entrenamiento.
        epochs: Número de épocas de entrenamiento.

    Returns:
        None
    """

    labels = label_gen(n_classes=len(common.LABELS_LIST))
    predictions = model([test_input, labels], training=False)
    plt.figure(figsize=(8, 2))

    print("Generated Images are Conditioned on Label:",
          common.class_map[np.array(labels)[0]])
    for i in range(predictions.shape[0]):
        pred = (predictions[i, :, :, :] + 1) * 127.5
        pred = np.array(pred)
        plt.subplot(1, 5, i+1)
        plt.imshow(pred.astype(np.uint8), cmap='gray')
        plt.axis('off')

    plt.savefig(common.IMAGE_EPOCHS_LOCATION +
                '/image_at_epoch_{:d}.png'.format(epoch))


def train(dataset, epochs):
    """
    Realiza el entrenamiento del modelo generativo condicional.

    Parameters:
        dataset: Conjunto de datos de entrenamiento.
        epochs: Número de épocas de entrenamiento.

    Returns:
        None
    """

    print('Iniciando entrenamiento...')

    for epoch in range(epochs):
        start = time.time()
        i = 0
        # D_loss_list, G_loss_list = [], []
        for image_batch, target in dataset:
            i += 1
            train_step(image_batch, target)
        print(epoch)
        generate_and_save_images(conditional_gen, epoch + 1, seed)

        conditional_gen.save_weights(
            common.EPOCHS_LOCATION + '/gen_' + str(epoch)+'.keras')
        conditional_disc.save_weights(
            common.EPOCHS_LOCATION + '/disc_' + str(epoch)+'.keras')
        print('Time for epoch {} is {} sec'.format(
            epoch + 1, time.time()-start))

    generate_and_save_images(conditional_gen, epochs, seed)


def label_gen(n_classes):
    """
    Genera una etiqueta aleatoria a partir del número de clases.

    Parameters:
        n_classes: Número de clases posibles.

    Returns:
        lab: Etiqueta generada.
    """

    lab = tf.random.uniform((1,), minval=0, maxval=n_classes,
                            dtype=tf.dtypes.int32, seed=None, name=None)
    return tf.repeat(lab, [NUM_EXAMPLES_TO_GENERATE], axis=None, name=None)
