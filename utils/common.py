CLASSES = [
    'mockup login webpage',
    'mockup profile webpage',
    'mockup e-mail webpage',
    'mockup chat webpage',
    'mockup blog webpage',
    # 'Mockup Board webpage',
    # 'Mockup E-Commerce webpage',
    # 'Mockup Shopping Cart webpage',
    # 'Mockup Control Panel webpage',
    # 'Mockup Calendar webpage',
    # 'Mockup News webpage',
    # 'Mockup Learning Platform webpage',
    # 'Mockup Travel Platform webpage',
    # 'Mockup Streaming Platform webpage',
    # 'Mockup Meeting Platform webpage',
    # 'Mockup ToDo List webpage',
    # 'Mockup Galery webpage',
    # 'Mockup Survey webpage',
    # 'Mockup Drawing webpage',
    # 'Mockup Search webpage',
]

BASE_CLASS = [
    'login',
    'profile',
    'e-mail',
    'chat',
    'blog'
]

KEYWORDS = {
    'login': ['authentication', 'access', 'identification', 'session'],
    'profile': ['personal', 'user', 'biography', 'main'],
    'e-mail': ['email', 'inbox', 'send', 'message', 'imbox', 'e-mail'],
    'chat': ['message', 'chatting'],
    'blog': ['journal', 'diary', 'weblog', 'e-diary'],
}

GAN_MODEL_LOCATION = './model/config'
IMAGES_LOCATION = './files/dataset/images'
WORD_MODEL_LOCATION = './files'
BIN_LOCATION = './files/dataset/binary'
EPOCHS_LOCATION = './model/weights_train'
IMAGE_EPOCHS_LOCATION = './model/epoch_images'
RESULTS_LOCATION = './model/results'
ALLOWED_EXTENSIONS = ('.jpg', '.jpeg')

GENERATOR_RES_FACTOR = 4  # (1=32, 2=64, 3=96, 4=128 ...)
GENERATE_SQUARE = 32 * GENERATOR_RES_FACTOR
CHANNELS = 3
LATENT_DIM = 100
EMBED_SIZE = 100
EPOCHS = 150
BATCH_SIZE = 16
BUFFER_SIZE = 10000
