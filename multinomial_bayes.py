import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from kaggle_preprocessing import getTarget, getTrain_Tfidf_Data


def naive_bayes(trainX, trainy):
    """ 
    使用 naive_bayes 模式训练学习器

    Parameter:
    ----------
    trainX : 训练数据
    trainy : 对应的标签

    Return:
    -------
    已训练好的学习器
    """

    clf = MultinomialNB()

    clf.fit(trainX, trainy)

    return clf


def preformance():
    X = getTrain_Tfidf_Data()
    y = getTarget()

    trainX, testX, trainy, testy = train_test_split(
        X, y, shuffle=True, test_size=0.2, random_state=42)

    print("开始训练")
    clf = naive_bayes(trainX, trainy)

    ypred = clf.predict(testX)

    score = f1_score(testy, ypred, average='micro')

    print("score = %.5f" % (score))


if __name__ == "__main__":
    preformance()
