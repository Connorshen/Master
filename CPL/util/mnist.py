# -*- coding: utf-8 -*-
"""
@Time    : 2019-08-12 14:30
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import numpy as np
from struct import unpack
from url import DataConfig


def __read_image(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img


def __read_label(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab


def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img


def __one_hot_label(label):
    lab = np.zeros((label.size, 10), dtype=np.uint8)
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab


def load_mnist(train_image_path, train_label_path, test_image_path, test_label_path, normalize=True, one_hot=True):
    image = {
        'train': __read_image(train_image_path),
        'test': __read_image(test_image_path)
    }

    label = {
        'train': __read_label(train_label_path),
        'test': __read_label(test_label_path)
    }

    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])

    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])

    return image['train'], label['train'], image['test'], label['test']


if __name__ == '__main__':
    train_image, train_label, test_image, test_label = load_mnist(DataConfig.TRAIN_IMAGE_PATH,
                                                                  DataConfig.TRAIN_LABEL_PATH,
                                                                  DataConfig.TEST_IMAGE_PATH,
                                                                  DataConfig.TEST_LABEL_PATH,
                                                                  normalize=True,
                                                                  one_hot=True)
