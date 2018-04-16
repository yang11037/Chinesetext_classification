#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import jieba

from rwTool import writefile, readfile, delfile


# 对有标签的数据集的分词方法
def corpus_seg(corpus_path, seg_path):
    '''corpus_path为初始数据集的路径
       seg_path为分类后要保存的路径
    '''
    type_list = os.listdir(corpus_path)    # 获取corpus_path下所有子目录，每个子目录代表一种类别

    print("正在执行分词操作")
    check = 0

    delfile(seg_path)  # 每次分词前删除上一次分词的残留

    for type_dir in type_list:
        type_path = corpus_path + type_dir + "/"  # typepath指每一个类别的路径
        save_path = seg_path + type_dir + "/"  # 分词后要存储的路径

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_list = os.listdir(type_path)  # 获取corpus_dir下所有子目录，即所有文件的名字

        for file_dir in file_list: # 遍历typedir目录下的所有文件
            file_path = type_path + file_dir  # 获取打开文件时的路径
            #save_path = save_path + file_dir
            content = readfile(file_path)

            # 下面开始做分词操作
            content = content.replace('\r\n'.encode('utf-8'), ''.encode('utf-8')).strip()
            content = content.replace(' '.encode('utf-8'), ''.encode('utf-8')).strip()
            '''以上消除了空行、空格、以及换行
               使分词后的结果更简洁
            '''
            content_seg = jieba.cut(content) # 使用jieba的精确模式分词
            writefile(save_path+file_dir, ' '.join(content_seg).encode('utf-8'))  #分词后的结果保存
        check = check + 1
        print("已分好", check, "个类别")

    print("分词结束")


# 对没有标签的文件进行实际分类
def corpus_segment2(corpus_path):
    file_list = os.listdir(corpus_path)
    seg_path = "./test_corpus_seg/"
    delfile(seg_path)
    print("正在执行分词操作...")

    for file_dir in file_list:
        file_path = corpus_path + "/" + file_dir
        content = readfile(file_path)

        # 下面开始做分词操作
        content = content.replace('\r\n'.encode('utf-8'), ''.encode('utf-8')).strip()
        content = content.replace(' '.encode('utf-8'), ''.encode('utf-8')).strip()
        '''以上消除了空行、空格、以及换行
           使分词后的结果更简洁
        '''
        content_seg = jieba.cut(content)  # 使用jieba的精确模式分词
        writefile(seg_path+file_dir, ' '.join(content_seg).encode('utf-8'))
    print("分词成功")


def corpus_segment3(file_path):
    seg_path="./test_corpus_seg"
    delfile(seg_path)
    print("正在执行分词操作...")

    content = readfile(file_path)

    # 下面开始做分词操作
    content = content.replace('\r\n'.encode('utf-8'), ''.encode('utf-8')).strip()
    content = content.replace(' '.encode('utf-8'), ''.encode('utf-8')).strip()
    '''以上消除了空行、空格、以及换行
       使分词后的结果更简洁
    '''
    content_seg = jieba.cut(content)  # 使用jieba的精确模式分词
    writefile(file_path, ' '.join(content_seg).encode('utf-8'))
    print("分词成功！")



def segment_train(path):
    # 对训练集分词
    # corpus_path = "./train_corpus/"
    corpus_path = path+"/"
    seg_path = "./train_corpus_seg/"
    # seg_path = path + "_seg/"
    corpus_seg(corpus_path, seg_path)
    print("训练集分词完成")


def segment_test(path, check=None):
    # 对测试集分词
    #corpus_path = "./test_corpus/"
    corpus_path = path+"/"
    seg_path = "./test_corpus_seg/"
    # seg_path = path + "_seg/"
    if check == 1:
        corpus_segment2(corpus_path)
    elif check == 2:
        corpus_path = "./Tools/cache.txt"
        corpus_segment3(corpus_path)
    else:
        corpus_seg(corpus_path, seg_path)
    print("测试集分词完成")


