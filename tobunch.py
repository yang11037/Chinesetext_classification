#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os

from sklearn.datasets.base import Bunch
from rwTool import readfile, writebunch


def tobunch(seg_path, word_bag_path):
    print("正在构建文本对象")
    type_list = os.listdir(seg_path)
    # 创建Bunch对象
    bunch = Bunch(type=[], label=[], filepath=[], filename=[],  content=[])
    '''
    type指类别，与label不同的是，type中的数据不会重复
    '''
    bunch.type.extend(type_list)  # 用type_list去填充type

    # 接下来遍历每个类别下的文件，将其构造成Bunch对象
    for type_dir in type_list:
        type_path = seg_path + type_dir + "/"
        file_list = os.listdir(type_path)
        for file_dir in file_list:
            filepath = type_path + file_dir
            # 将文件的路径、类别、内容赋值给Bunch对象
            bunch.filepath.append(filepath)
            bunch.filename.append(file_dir)
            bunch.content.append(readfile(filepath))
            bunch.label.append(type_dir)
    # 将Bunch对象写入word_bag_path中
    writebunch(word_bag_path, bunch)


def tobunch2(seg_path, word_bag_path):
    print("正在构建文本对象")
    bunch = Bunch(type=[], label=[], filepath=[], filename=[], content=[])

    file_list = os.listdir(seg_path)
    for file_dir in file_list:
        filepath = seg_path + file_dir
        bunch.filepath.append(filepath)
        bunch.filename.append(file_dir)
        bunch.content.append(readfile(filepath))

    writebunch(word_bag_path, bunch)


def tobunch3(seg_path, word_bag_path):
    print("正在构建文本对象")
    bunch = Bunch(type=[], label=[], filepath=[], filename=[], content=[])

    bunch.filepath.append(seg_path)
    bunch.content.append(readfile(seg_path))
    writebunch(word_bag_path, bunch)


def build_Bunch_train():
    # 对训练集
    word_bag_path = "./train_corpus_bag/train.dat"
    seg_path = "./train_corpus_seg/"
    tobunch(seg_path, word_bag_path)
    print("训练集构建Bunch成功")


def build_Bunch_test(check=None):
    # 对测试集
    word_bag_path = "./test_corpus_bag/test.dat"
    seg_path = "./test_corpus_seg/"
    if check == 1:
        tobunch2(seg_path, word_bag_path)
    elif check == 2:
        seg_path = "./Tools/cache.txt"
        tobunch3(seg_path, word_bag_path)
    else:
        tobunch(seg_path, word_bag_path)
    print("测试集构建Bunch成功")