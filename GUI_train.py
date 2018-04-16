#!/usr/bin/env python
# -*- coding:utf-8 -*-


from tkinter import *
from naive_Bayes import NB
from corpus_seg import *
from tobunch import *
from tf_idf import *
from tkinter.filedialog import askdirectory
from Knn import *
from pre_test import *
from DTree import *


def selectPath1():
    path_ = askdirectory()
    path1.set(path_)

def selectPath2():
    path_ = askdirectory()
    path2.set(path_)


def NBcheck():
    segment_train(path1.get())
    build_Bunch_train()
    build_tfidf_train()
    NB(path1.get())
    text.insert(END, "训练完成！\n")



def Knncheck():
    segment_train(path1.get())
    build_Bunch_train()
    build_tfidf_train()
    Knn_train(path1.get())
    text.insert(END, "训练完成！\n")


def DTcheck():
    segment_train(path1.get())
    build_Bunch_train()
    build_tfidf_train()
    DTree_train(path1.get())
    text.insert(END, "训练完成！\n")


def test():
    segment_test(path2.get())
    build_Bunch_test()
    build_tfidf_test()
    a, sum, check = pre_test()
    text.insert(END, "分类完成！\n")
    text.insert(END, "测试集总数为："+str(sum)+"\n")
    text.insert(END, "分类错误数为："+str(check)+"\n")
    text.insert(END, "分类精度为："+a + "\n")


root = Tk()
root.title("分类器")
root.iconbitmap('./Tools/crystal.ico')


width, height = 390, 270
root.geometry('%dx%d+%d+%d' % (width,height, (root.winfo_screenwidth() - width)
                               / 2, (root.winfo_screenheight() - height) / 2))


root.maxsize(390, 270)
root.minsize(390, 270)

path1 = StringVar()
path2 = StringVar()

text = Text(root, height=10, width=40)
text.grid(row=0, rowspan=3)
text.insert(1.0,"等待操作\n")


Button(root, text="Knn训练器", command=Knncheck).grid(row=0, column=1)
Button(root, text="决策树训练器", command=DTcheck).grid(row=1, column=1)
Button(root, text="贝叶斯训练器", command=NBcheck).grid(row=2, column=1)
Button(root, text="训练集路径", command=selectPath1).grid(row=4, column=1)
e1 = Entry(root, textvariable=path1, width=35).grid(row=4, column=0)
Button(root, text="测试集路径", command=selectPath2).grid(row=6, column=1)
e2 = Entry(root, textvariable=path2, width=35).grid(row=6, column=0)
Button(root, text="测试分类器精度", command=test).grid(row=8, columnspan=2)

root.mainloop()