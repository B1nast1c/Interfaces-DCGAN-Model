import os
from utils import common


def create_all_captions():
    create_captions()


def create_captions():
    for class_index in range(len(common.BASE_CLASS)):
        file_name = common.BASE_CLASS[class_index]
        tags = common.KEYWORDS[class_index]
        sentences = ['a ' + tag +
                     ' webpage interface' for tag in tags]

        location = os.path.join(
            common.CAPTIONS_LOCATION, f'{file_name}.txt')
        with open(location, 'w+') as file:
            for sentence in sentences:
                file.write('%s\n' % sentence)
