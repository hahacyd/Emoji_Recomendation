from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split,cross_val_score
from labelencode import KaggleLabelEncode
import time
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords
def fit(X, y):
    # clf = MultinomialNB()
    clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)
    clf.fit(X, y)
    
    # joblib.dump(clf, "model/dump/NaiveBayes_emotion.pkl")
    joblib.dump(clf,"model/dump/svm_eomtion.pkl")
    return clf
def load_clf(path):
    clf = joblib.load(path)
    return clf
def main():
    file = open("model/train.csv")
    print("加载数据完成")
    X = joblib.load("model/dump/X.data")
    
    print(X.shape)

    y = joblib.load("model/dump/y.data")
    print("数据预处理完成")
    
    file.close()
    # clf = MultinomialNB()

    print("开始训练")
    time_start = time.time()

    # clf = SGDClassifier(alpha=1e-3,random_state=42, n_jobs=2,max_iter=5)
    # clf = MultinomialNB()
    # clf = LinearSVC(dual=False, verbose=1)
    # clf = DecisionTreeClassifier()
     
    # clf.fit(X, y)
    # joblib.dump(clf,"model/dump/svm_emotion.pkl")
    scores=cross_val_score(clf, X, y, cv=5, verbose=1)
    
    time_end=time.time()
    
    print(scores)
    # clf = load_clf("model/dump/svm_emotion.pkl")
    # print("训练完成")
    print("测试完成 , 用时 %.5f 秒"%(time_end - time_start))

if __name__ == "__main__":
    main()
