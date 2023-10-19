import os
import time
from model import generator, discriminator
import numpy as np
import tensorflow as tf
import test_data_processing
from utils import common, split_data
from keras.optimizers import Adam

cross_entropy = tf.keras.losses.BinaryCrossentropy()
save_images_npy, save_images_captions, save_images_embeddings = test_data_processing.test_data_splitting()
channels = common.CHANNELS
image_shape = (common.GENERATE_SQUARE, common.GENERATE_SQUARE, channels)

generator_optimizer = Adam(learning_rate=2.0e-4, beta_1=0.5)
discriminator_optimizer = Adam(learning_rate=2.0e-4, beta_1=0.5)

generator_item = generator.create_generator(
    common.SEED_SIZE, common.EMBED_SIZE, channels)
discriminator_item = discriminator.create_discriminator(
    image_shape, common.EMBED_SIZE)


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_image_real_text, fake_image_real_text, real_image_fake_text):
    real_loss = cross_entropy(tf.random.uniform(
        real_image_real_text.shape, 0.8, 1.0), real_image_real_text)
    fake_loss = (cross_entropy(tf.random.uniform(fake_image_real_text.shape, 0.0, 0.2), fake_image_real_text) +
                 cross_entropy(tf.random.uniform(real_image_fake_text.shape, 0.0, 0.2), real_image_fake_text))/2

    total_loss = real_loss + fake_loss
    return total_loss


@tf.function
def train_step(images, captions, fake_captions):
    seed = tf.random.normal(
        [common.BATCH_SIZE, common.SEED_SIZE], dtype=tf.float32)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator_item((seed, captions), training=True)
        real_image_real_text = discriminator_item(
            (images, captions), training=True)
        real_image_fake_text = discriminator_item(
            (images, fake_captions), training=True)
        fake_image_real_text = discriminator_item(
            (generated_images, captions), training=True)

        gen_loss = generator_loss(fake_image_real_text)
        disc_loss = discriminator_loss(
            real_image_real_text, fake_image_real_text, real_image_fake_text)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator_item.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator_item.trainable_variables)

        generator_optimizer.apply_gradients(zip(
            gradients_of_generator, generator_item.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(
            gradients_of_discriminator,
            discriminator_item.trainable_variables))
    return gen_loss, disc_loss


def train(train_dataset, epochs):
    ''' fixed_seed = np.random.normal(0, 1, (common.PREVIEW_ROWS * common.PREVIEW_COLS,
                                        common.SEED_SIZE))
    fixed_embed = save_images_embeddings '''

    start = time.time()

    for epoch in range(epochs):
        print("Epoch start...")
        epoch_start = time.time()

        gen_loss_list = []
        disc_loss_list = []

        for batch in train_dataset[:-1]:
            train_batch = batch['images']
            caption_batch = batch['embeddings']

            fake_caption_batch = np.copy(caption_batch)
            np.random.shuffle(fake_caption_batch)

            t = train_step(train_batch, caption_batch, fake_caption_batch)
            gen_loss_list.append(t[0])
            disc_loss_list.append(t[1])

        g_loss = sum(gen_loss_list) / len(gen_loss_list)
        d_loss = sum(disc_loss_list) / len(disc_loss_list)

        epoch_elapsed = time.time()-epoch_start
        print(
            f'Epoch {epoch+1}, gen loss={g_loss},disc loss={d_loss}, {common.time_shower(epoch_elapsed)}')
        # save_images(epoch,fixed_seed,fixed_embed)

        generator_item.save(os.path.join(
            common.GAN_MODEL_LOCATION, "generator.keras"))
        discriminator_item.save(os.path.join(
            common.GAN_MODEL_LOCATION, "discriminator.keras"))

    elapsed = time.time()-start
    print('Training time:', common.time_shower(elapsed))


def train_gan():
    train_dataset = split_data.shuffle_data()
    train(list(train_dataset.as_numpy_iterator()), 50)
