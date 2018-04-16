#!/usr/bin/env python
# -*- coding:utf-8 -*-


from sklearn import tree  # 导入决策树
from rwTool import readbunch
from sklearn.externals import joblib


def DTree_train(path):
    # 导入训练集
    trainpath = "./train_corpus_bag/train_tfidf.dat"
    train_set = readbunch(trainpath)

    # 训练分类器：输入词袋向量和分类标签，alpha:0.001 alpha越小，迭代次数越多，精度越高
    clf = tree.DecisionTreeClassifier(criterion='entropy').fit(train_set.tfidf, train_set.label)

    joblib.dump(clf, "./Tools/train_clf.pkl")

    print("训练完成")