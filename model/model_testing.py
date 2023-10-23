import os
import tensorflow as tf
import numpy as np
from model import generator, discriminator
from utils import text_model, common
from PIL import Image

channels = common.CHANNELS
image_shape = (common.GENERATE_SQUARE, common.GENERATE_SQUARE, channels)
weights_location = common.GAN_MODEL_LOCATION

generator_item = generator.create_generator(
    common.SEED_SIZE, common.EMBED_SIZE, channels)
discriminator_item = discriminator.create_discriminator(
    image_shape, common.EMBED_SIZE)

generator_item.load_weights(weights_location + '/generator.keras')
discriminator_item.load_weights(weights_location + '/discriminator.keras')


def save_images(cnt, noise, embeds):
    image_array = np.full((
        common.PREVIEW_MARGIN +
        (common.PREVIEW_ROWS * (common.GENERATE_SQUARE+common.PREVIEW_MARGIN)),
        common.PREVIEW_MARGIN + (common.PREVIEW_COLS * (common.GENERATE_SQUARE+common.PREVIEW_MARGIN)), 3),
        255, dtype=np.uint8)

    generated_images = generator_item.predict((noise, embeds))

    generated_images = 0.5 * generated_images + 0.5

    image_count = 0
    for row in range(common.PREVIEW_ROWS):
        for col in range(common.PREVIEW_COLS):
            r = row * (common.GENERATE_SQUARE+5) + common.PREVIEW_MARGIN
            c = col * (common.GENERATE_SQUARE+5) + common.PREVIEW_MARGIN
            image_array[r:r+common.GENERATE_SQUARE, c:c+common.GENERATE_SQUARE] \
                = generated_images[image_count] * 255
            image_count += 1

    output_path = common.RESULTS_LOCATION
    filename = os.path.join(output_path, f"test{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)


def test_gan(text, num):
    test_embeddings = np.zeros((1, 300), dtype=np.float32)
    word2vec_embeddings = text_model.load_model()

    x = text.lower()
    x = x.split()
    count = 0

    for t in x:
        try:
            test_embeddings[0] += word2vec_embeddings[t]
            count += 1
        except Exception as ex:
            print(ex)
            pass

    test_embeddings[0] /= count
    test_embeddings = np.repeat(test_embeddings, [3], axis=0)
    noise = tf.random.normal([3, 100])

    save_images(num, noise, test_embeddings)
