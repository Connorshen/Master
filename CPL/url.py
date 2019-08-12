# -*- coding: utf-8 -*-
"""
@Time    : 2019-08-12 15:37
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
from os import path


class DataConfig:
    ROOT = path.dirname(path.realpath(__file__))
    DATA_PATH = path.join(ROOT, "data")
    TRAIN_IMAGE_PATH = path.join(DATA_PATH, "train-images-idx3-ubyte")
    TRAIN_LABEL_PATH = path.join(DATA_PATH, "train-labels-idx1-ubyte")
    TEST_IMAGE_PATH = path.join(DATA_PATH, "t10k-images-idx3-ubyte")
    TEST_LABEL_PATH = path.join(DATA_PATH, "t10k-labels-idx1-ubyte")
