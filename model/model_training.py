import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import generator, discriminator
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from utils import common

# PARAMETROS
# ----------------------------------------------------------------

conditional_gen = generator.gan_generator()
conditional_disc = discriminator.gan_discriminator()
num_examples_to_generate = 5
generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
seed = tf.random.normal([num_examples_to_generate, common.LATENT_DIM])
binary_cross_entropy = BinaryCrossentropy()


def generator_loss(label, fake_output):
    gen_loss = binary_cross_entropy(label, fake_output)
    return gen_loss


def discriminator_loss(label, output):
    disc_loss = binary_cross_entropy(label, output)
    return disc_loss

# ----------------------------------------------------------------


@tf.function
def train_step(images, target):
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
    labels = label_gen(n_classes=len(common.BASE_CLASS))
    predictions = model([test_input, labels], training=False)
    # fig = plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        pred = (predictions[i, :, :, :] + 1) * 127.5
        pred = np.array(pred)

    plt.savefig(common.IMAGE_EPOCHS_LOCATION +
                '/image_at_epoch_{:d}.png'.format(epoch))


def train(dataset, epochs):
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
    lab = tf.random.uniform((1,), minval=0, maxval=n_classes,
                            dtype=tf.dtypes.int32, seed=None, name=None)
    return tf.repeat(lab, [num_examples_to_generate], axis=None, name=None)
