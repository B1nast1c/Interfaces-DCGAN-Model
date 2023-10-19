from utils import create_csv, create_captions, text_model, process_captions, process_images, load_bin, split_data

# Procesamiento de datos
create_csv.vocab_csv()
create_captions.create_all_captions()
text_model.create_model()
process_images.save_images()
process_captions.save_captions()
