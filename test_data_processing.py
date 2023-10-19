from utils import process_captions, process_images

# Obtención de training_data y testing_data


def test_data_splitting():
    save_images_npy, indexes = process_images.get_test_images()
    save_images_captions = process_captions.get_test_captions(indexes)
    save_images_embeddings = process_captions.get_test_embeddings(indexes)

    return save_images_npy, save_images_captions, save_images_embeddings
