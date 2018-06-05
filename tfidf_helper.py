#!/usr/bin/env python
# -*- coding:utf-8 -*-


from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from rwTool import readfile, readbunch, writebunch


class TfidfHelper:
    def __init__(self):
        pass

    @staticmethod
    def tf_idf(stpwrdlst_path, bunch_path, space_path, tfidf_path=None):
        stpwrdlst = readfile(stpwrdlst_path).splitlines()  # 读取停用词表
        bunch = readbunch(bunch_path)  # 读取Bunch对象
        tfidf_bunch = Bunch(label=bunch.label, filepath=bunch.filepath, filename=bunch.filename,
                            content=bunch.content, tfidf=[], vocabSet=[])
        # 构建了一个用于存储词向量空间的Bunch对象

        # 若用于构建训练集空间
        if tfidf_path is None:
            vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)
            '''
            初始化向量空间模型
            sublinear指用1+log(tf)来代替tf
            max_df指df的值在[0,0.5]之间,以删除某些出现次数过多，无分类意义的词干扰
            '''
            tfidf_bunch.tfidf = vectorizer.fit_transform(tfidf_bunch.content)
            # 构建TF-IDF权重矩阵
            tfidf_bunch.vocabSet = vectorizer.vocabulary_  # 将词频矩阵赋值给tfidf对象

        # 若用于构建测试集空间
        else:
            # 构建测试集空间时需导入训练集词向量空间，使训练集空间与测试集一致
            train_bunch = readbunch(tfidf_path)
            tfidf_bunch.vocabSet = train_bunch.vocabSet
            vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,
                                         vocabulary=tfidf_bunch.vocabSet)
            tfidf_bunch.tfidf = vectorizer.fit_transform(tfidf_bunch.content)

        writebunch(space_path, tfidf_bunch)

        if tfidf_path is None:
            print("构建训练集tf-idf词向量空间成功！")
        else:
            print("构建测试集tf-idf词向量空间成功！")



    def build_tfidf_train(self):
        # 对训练集构建空间
        stpwrdlst_path = "./Tools/stopword.txt"
        bunch_path = "./train_corpus_bag/train.dat"
        space_path = "./train_corpus_bag/train_tfidf.dat"
        self.tf_idf(stpwrdlst_path, bunch_path, space_path)


    def build_tfidf_test(self):
        # 对测试集构建空间
        stpwrdlst_path = "./Tools/stopword.txt"
        bunch_path = "./test_corpus_bag/test.dat"
        space_path = "./test_corpus_bag/test_tfidf.dat"
        tfidf_path = "./train_corpus_bag/train_tfidf.dat"
        self.tf_idf(stpwrdlst_path, bunch_path, space_path, tfidf_path) # 测试集的词向量必须与训练集相同

