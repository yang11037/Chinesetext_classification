#!/usr/bin/env python
# -*- coding:utf-8 -*-


from tkinter import *
from tkinter.filedialog import askdirectory
from seg_helper import SegHelper
from bunch_helper import *
from tfidf_helper import *
from sklearn.externals import joblib
from rwTool import movefile, writefile

def selectpath():
    path_ = askdirectory()
    path.set(path_)


def start():
    # 选择路径时
    if var.get() == 0:
        seg = SegHelper(path.get(), 1)
        seg.segment_test()

        bunchhelper = BunchHelper(1)
        bunchhelper.build_Bunch_test()

        Tfidf = TfidfHelper()
        Tfidf.build_tfidf_test()
        # 导入测试集

        testpath = "test_corpus_bag/test_tfidf.dat"
        test_set = readbunch(testpath)

        # 预测分类结果
        clf = joblib.load("./Tools/train_clf.pkl")
        result = clf.predict(test_set.tfidf)
        for filename, filepath, expectation in zip(test_set.filename, test_set.filepath, result):
            type_path = path.get() + "/" + expectation + "/"
            if not os.path.exists(type_path):
                os.makedirs(type_path)
            des_path = type_path + filename
            movefile(filepath, des_path)
        text.insert(END, "分类成功！\n")

    # 手动输入时
    elif var.get() == 1:
        txt_path = "./Tools/cache.txt"  # 设置一个缓存文本
        if os.path.exists(txt_path):
            os.remove(txt_path)      # 每次都将上一次的缓存删除
        content = path.get().encode('utf-8')
        writefile(txt_path, content)
        seg = SegHelper(txt_path, 2)
        seg.segment_test()

        bunchhelper = BunchHelper(2)
        bunchhelper.build_Bunch_test()

        Tfidf = TfidfHelper()
        Tfidf.build_tfidf_test()

        testpath = "test_corpus_bag/test_tfidf.dat"
        test_set = readbunch(testpath)
        clf = joblib.load("./Tools/train_clf.pkl")
        result = clf.predict(test_set.tfidf)
        for expectation in zip(result):
            text.insert(END, "预测成功，您输入的文本属于" + str(expectation) + "\n")




root = Tk()
root.title("分类器")
root.iconbitmap('./Tools/crystal.ico')

width, height = 390, 250
root.geometry('%dx%d+%d+%d' % (width,height, (root.winfo_screenwidth() - width)
                               / 2, (root.winfo_screenheight() - height) / 2))


root.maxsize(390, 250)
root.minsize(390, 250)

text = Text(root, height=10, width=54)
text.grid(row=0, rowspan=3, columnspan=2)
text.insert(1.0, "等待操作\n")

path = StringVar()
var = IntVar()

e1 = Entry(root, textvariable=path, width=35).grid(row=5, column=0)
Button(root, text="路径选择", command=selectpath).grid(row=5, column=1)
Radiobutton(root, text="手动输入", value=1, variable=var).grid(row=7, column=0)
Radiobutton(root, text="选择路径", value=0, variable=var).grid(row=7, column=1)
Button(root, text="开始分类", command=start).grid(row=11, columnspan=2)


root.mainloop()