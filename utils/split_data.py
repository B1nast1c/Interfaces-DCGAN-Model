from utils import load_bin, common
from tensorflow import data
from utils import common


def shuffle_data():
    final_images = load_bin.load_images()
    final_embeddings = load_bin.load_captions()
    # Batch and shuffle the data
    train_dataset = data.Dataset.from_tensor_slices({'images': final_images,
                                                     'embeddings': final_embeddings}).shuffle(common.BUFFER_SIZE).batch(common.BATCH_SIZE)
    return train_dataset
