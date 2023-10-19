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

KEYWORDS = [
    ['login', 'authentication', 'access', 'identification', 'session'],
    ['profile', 'personal', 'user', 'biography', 'main'],
    ['email', 'inbox', 'send', 'message', 'imbox', 'e-mail'],
    ['chat', 'message', 'chatting'],
    ['blog', 'journal', 'diary', 'weblog', 'e-diary'],
]

GAN_MODEL_LOCATION = './model/config'
IMAGES_LOCATION = './files/dataset/images'
CAPTIONS_LOCATION = './files/captions'
SINGLE_CAPTIONS_LOCATION = './files/dataset/captions'
WORD_MODEL_LOCATION = './files'
BIN_LOCATION = './files/dataset/binary'
ALLOWED_EXTENSIONS = ('.jpg', '.jpeg')

GENERATOR_RES_FACTOR = 4  # (1=32, 2=64, 3=96, 4=128 ...)
GENERATE_SQUARE = 32 * GENERATOR_RES_FACTOR
CHANNELS = 3
SEED_SIZE = 100
EMBED_SIZE = 300
EPOCHS = 50
BATCH_SIZE = 16
BUFFER_SIZE = 512
WIDTH = 480
HEIGHT = 270
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16


def time_shower(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)
