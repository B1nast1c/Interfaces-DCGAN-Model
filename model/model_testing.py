import tensorflow as tf
import numpy as np
from utils import text_model


def test_gan(text, num):
    test_embeddings = np.zeros((1, 300), dtype=np.float32)
    word2vec_embeddings = text_model.load_model()

    x = text.lower()
    x = x.replace(" ", "")
    count = 0
    for t in x:
        try:
            test_embeddings[0] += word2vec_embeddings[t]
            count += 1
        except:
            print(t)
            pass
    test_embeddings[0] /= count
    test_embeddings = np.repeat(test_embeddings, [28], axis=0)
    noise = tf.random.normal([28, 100])
    # save_images(num, noise, test_embeddings)
