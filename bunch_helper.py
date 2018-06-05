#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os

from sklearn.datasets.base import Bunch
from rwTool import readfile, writebunch


class BunchHelper:
    def __init__(self, check = None):
        self.check = check

    # 对已经有类别标签的数据集
    @staticmethod
    def tobunch(seg_path, word_bag_path):
        print("正在构建文本对象")
        type_list = os.listdir(seg_path)
        # 创建Bunch对象
        bunch = Bunch(label=[], filepath=[], filename=[],  content=[])

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


    # 对无类别标签的数据集
    @staticmethod
    def tobunch2(seg_path, word_bag_path):
        print("正在构建文本对象")
        bunch = Bunch(label=[], filepath=[], filename=[], content=[])

        file_list = os.listdir(seg_path)
        for file_dir in file_list:
            filepath = seg_path + file_dir
            bunch.filepath.append(filepath)
            bunch.filename.append(file_dir)
            bunch.content.append(readfile(filepath))

        writebunch(word_bag_path, bunch)

    # 对输入框的数据
    @staticmethod
    def tobunch3(seg_path, word_bag_path):
        print("正在构建文本对象")
        bunch = Bunch(label=[], filepath=[], filename=[], content=[])

        bunch.filepath.append(seg_path)
        bunch.content.append(readfile(seg_path))
        writebunch(word_bag_path, bunch)


    def build_Bunch_train(self):
        # 对训练集
        word_bag_path = "./train_corpus_bag/train.dat"
        seg_path = "./train_corpus_seg/"
        self.tobunch(seg_path, word_bag_path)
        print("训练集构建Bunch成功")


    def build_Bunch_test(self):
        # 对测试集
        word_bag_path = "./test_corpus_bag/test.dat"
        seg_path = "./test_corpus_seg/"
        if self.check == 1:
            self.tobunch2(seg_path, word_bag_path)
        elif self.check == 2:
            seg_path = "./Tools/cache.txt"
            self.tobunch3(seg_path, word_bag_path)
        else:
            self.tobunch(seg_path, word_bag_path)
        print("测试集构建Bunch成功")