#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import pickle
import os
import shutil


# 写文档
def writefile(path, content):
    with open(path,"wb") as fp:
        fp.write(content)


# 读文档
def readfile(path):
    with open(path,"rb") as fp:
        content = fp.read()

    return content


# 写bunch对象
def writebunch(path, bunch_obj):
    with open(path, "wb") as fp:
        pickle.dump(bunch_obj, fp)


# 读bunch对象
def readbunch(path):
    with open(path, "rb") as fp:
        bunch_obj = pickle.load(fp);
    return bunch_obj


# 删除文件
def delfile(path):
    delList = os.listdir(path)

    for f in delList:
        filepath = os.path.join(path, f)
        if os.path.isfile(filepath):
            os.remove(filepath)
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath, True)


# 移动文件
def movefile(src, des):
    content = readfile(src)
    writefile(des, content)
