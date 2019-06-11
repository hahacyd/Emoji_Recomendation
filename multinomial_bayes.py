from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import f1_score,make_scorer
def fit():
    X = joblib.load("model/dump/X.data")
    y = joblib.load("model/dump/y.data")

    clf = MultinomialNB()
    print("开始训练")
    clf.fit(X, y)
    joblib.dump(clf, "model/dump/naive_bayes.model")
    print("完成 模型保存完毕")
def main():
    X = joblib.load("model/dump/X.data")
    y = joblib.load("model/dump/y.data")

    clf = MultinomialNB()
    
    # clf.fit(X,y)
    score = cross_val_score(clf, X, y, cv=5, verbose=1,scoring=make_scorer(f1_score,average='micro'))
    
    print(score)
if __name__ == "__main__":
    # fit()
    main()