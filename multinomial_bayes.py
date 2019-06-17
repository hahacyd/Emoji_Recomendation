from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC,SVC,NuSVC

from kaggle_preprocessing import getTrain_Tfidf_Data, getTarget


def svm(trainX,trainy):
    """ 
    使用 svm 模式训练学习器

    Parameter :
    ----------
    trainX : 训练数据
    trainy : 对应的标签

    Return :
    -------
    已训练好的学习器
    """    

    clf = OneVsOneClassifier(LinearSVC(verbose=1),n_jobs=3)
    # clf = OneVsOneClassifier(SVC(verbose=1))
    clf.fit(trainX, trainy)
    joblib.dump(clf,"dump/svm.model")
    return clf

def naive_bayes(trainX,trainy):
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

    trainX, testX, trainy, testy = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

    print("开始训练")
    print(type(X))
    # clf = naive_bayes(trainX, trainy)
    clf = svm(trainX,trainy)

    ypred = clf.predict(testX)

    score = f1_score(testy, ypred, average='micro')
    
    print("score = %.5f" % (score))
if __name__ == "__main__":
    preformance()