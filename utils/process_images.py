import os
from utils import common, load_bin
import numpy as np
import cv2

IMG_DIR_PATH = common.IMAGES_LOCATION
EXTENSION = common.ALLOWED_EXTENSIONS


def tup(point):
    return (point[0], point[1])


def overlap(source, target):
    tl1, br1 = source
    tl2, br2 = target

    if (tl1[0] >= br2[0] or tl2[0] >= br1[0]):
        return False
    if (tl1[1] >= br2[1] or tl2[1] >= br1[1]):
        return False
    return True


def getAllOverlaps(boxes, bounds, index):
    overlaps = []
    for box_idx in range(len(boxes)):
        if box_idx != index:
            if overlap(bounds, boxes[box_idx]):
                overlaps.append(box_idx)
    return overlaps


def medianCanny(image, thresh1, thresh2):
    median = np.median(image)
    image = cv2.Canny(image, int(thresh1 * median), int(thresh2 * median))
    return image


def extract_contour(image):
    image = cv2.imread(image)
    original = np.copy(image)
    blue, green, red = cv2.split(image)
    blue_edges = medianCanny(blue, 0, 1)
    green_edges = medianCanny(green, 0, 1)
    red_edges = medianCanny(red, 0, 1)
    edges = blue_edges | green_edges | red_edges
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    hierarchy = hierarchy[0]

    for component in zip(contours, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]
        x, y, width, height = cv2.boundingRect(currentContour)
        if currentHierarchy[3] < 0:
            boxes.append([[x, y], [x+width, y+height]])

    filtered = []
    max_area = 30000
    for box in boxes:
        width = box[1][0] - box[0][0]
        height = box[1][1] - box[0][1]
        if width*height < max_area:
            filtered.append(box)
    boxes = filtered

    merge_margin = 10
    finished = False

    while not finished:
        finished = True
        index = 0
        while index < len(boxes):
            current_box = boxes[index]
            top_left = current_box[0][:]
            bottom_right = current_box[1][:]
            top_left[0] -= merge_margin
            top_left[1] -= merge_margin
            bottom_right[0] += merge_margin
            bottom_right[1] += merge_margin

            overlaps = getAllOverlaps(boxes, [top_left, bottom_right], index)

            if len(overlaps) > 0:
                con = []
                overlaps.append(index)
                for ind in overlaps:
                    top_left, bottom_right = boxes[ind]
                    con.append([top_left])
                    con.append([bottom_right])
                con = np.array(con)

                x, y, width, height = cv2.boundingRect(con)
                width -= 1
                height -= 1
                merged = [[x, y], [x+width, y+height]]

                overlaps.sort(reverse=True)
                for ind in overlaps:
                    del boxes[ind]

                boxes.append(merged)
                finished = False
                break
            index += 1

    copy = np.ones_like(original) * 255

    for box in boxes:
        cv2.rectangle(copy, tup(box[0]), tup(box[1]), (0, 0, 0), 3)
    copy = cv2.resize(copy, dsize=(
        common.GENERATE_SQUARE, common.GENERATE_SQUARE))

    return copy

# Carga TODAS las imagenes


def load_and_scale_images():
    images = []
    files = os.listdir(IMG_DIR_PATH)
    for file in files:
        filepath = os.path.join(IMG_DIR_PATH, file)
        if os.path.isfile(filepath) and file.endswith(EXTENSION[0]):
            image = extract_contour(filepath)
            images.append(image)
    images = np.array(images)
    return images


def save_images():
    training_data = load_and_scale_images()
    np.save(common.BIN_LOCATION + '/images.npy',
            training_data, allow_pickle=True)


def get_images_dataset():
    counter = 0
    bin_images = load_bin.load_images()
    train_images_bin, test_images_bin = [], []
    indexes = []
    images = os.listdir(IMG_DIR_PATH)

    for single in common.BASE_CLASS:
        indexes.append(len([item for item in images if single in item]))

    for index in indexes:
        test_index = int(index*0.25) - 1
        test_images = bin_images[counter: counter + test_index]
        test_images_bin.extend(test_images)
        train_images = bin_images[counter + test_index: counter + index]
        train_images_bin.extend(train_images)
        counter += index

    test_images_bin = np.array(test_images_bin)

    train_images_bin = np.array(train_images_bin)
    train_images_bin = (train_images_bin.astype(np.float32) - 127.5) / 127.5
    train_images_bin = np.clip(train_images_bin, -1, 1)

    print(train_images_bin.shape)

    return train_images_bin, test_images_bin
