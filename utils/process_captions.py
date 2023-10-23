import os
import numpy as np
from utils import text_model, common, load_bin


def captions_per_image():
    captions = []
    files_location = os.path.join(common.IMAGES_LOCATION)
    images = os.listdir(files_location)
    caption_embeddings = np.zeros((len(images), 300), dtype=np.float32)
    text_embeddings = text_model.load_model()

    for image in range(len(images)):
        image_class = images[image].split('_')[1]
        captions_file = open(common.CAPTIONS_LOCATION +
                             '/' + image_class + '.txt', 'r')
        file_data = captions_file.read()
        file_data = file_data.split('\n')
        captions_file.close()

        for index in range(1):
            single_caption = file_data[index].lower()
            captions.append(single_caption)
            single_caption = single_caption.split()
            count = 0

            for word in single_caption:
                try:
                    caption_embeddings[image] += text_embeddings[word]
                    count += 1
                except Exception as ex:
                    print(ex)
                    pass

            caption_embeddings[image] /= count

        return caption_embeddings, captions


def save_captions():
    text_embeddings, _ = captions_per_image()
    np.save(common.BIN_LOCATION + '/captions.npy',
            text_embeddings, allow_pickle=True)


def get_test_embeddings(indexes):
    counter = 0
    bin_embed = load_bin.load_captions()
    test_embed_bin = []
    for index in indexes:
        q_images = int(index*0.25) - 1
        test_embed_bin.extend(bin_embed[counter:counter+q_images])
        counter += index

    test_embed_bin = np.array(test_embed_bin)
    return test_embed_bin


def get_test_captions(indexes):
    _, captions = captions_per_image()
    counter = 0
    captions_bin = []

    for index in indexes:
        q_images = int(index*0.25) - 1
        captions_bin.extend(captions[counter:counter+q_images])
        counter += index

    return captions_bin
