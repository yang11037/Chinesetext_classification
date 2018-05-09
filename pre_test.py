#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn import metrics
from rwTool import readbunch
from sklearn.externals import joblib


def metrics_result(actual, predict):
    print('精度:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
    print('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))
    return '{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')), \
           '{0:.3f}'.format(metrics.recall_score(actual, predict, average='weighted')), \
           '{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted'))


def pre_test():

    # 导入测试集

    testpath = "test_corpus_bag/test_tfidf.dat"
    test_set = readbunch(testpath)

    # 预测分类结果
    clf = joblib.load("./Tools/train_clf.pkl")
    result = clf.predict(test_set.tfidf)

    check = 0
    sum = 0

    for label, filepath, expectation in zip(test_set.label, test_set.filepath, result):
        sum = sum + 1
        if label != expectation:
            check = check + 1
            print(check, filepath, ": 实际类别:", label, " -->预测类别:", expectation)

    print("预测完毕!!!")

    # 计算分类精度：

    a, b, c = metrics_result(test_set.label, result)
    return a, b, c, sum, check