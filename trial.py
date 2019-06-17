from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
import time

def fit(X, y):
    # clf = MultinomialNB()
    clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)
    clf.fit(X, y)
    
    # joblib.dump(clf, "dump/NaiveBayes_emotion.pkl")
    joblib.dump(clf,"dump/svm_eomtion.pkl")
    return clf
def load_clf(path):
    clf = joblib.load(path)
    return clf
def main():
    file = open("train.csv")
    print("加载数据完成")
    X = joblib.load("dump/X.data")
    
    print(X.shape)

    y = joblib.load("dump/y.data")
    print("数据预处理完成")
    
    file.close()

    print("开始训练")
    # 产生训练集和验证集
    transformer = joblib.load("dump/tfidf_vec.vec")
    X = transformer.transform(X)
    trainX, testX, trainy, testy = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
    
    time_start = time.time()

    # clf = SGDClassifier(alpha=1e-3,random_state=42, n_jobs=2,max_iter=5)
    clf = MultinomialNB()
    # clf = LinearSVC(dual=False, verbose=1)
    # clf = DecisionTreeClassifier()
     
    clf.fit(trainX,trainy)
    # joblib.dump(clf,"dump/svm_emotion.pkl")
    # scores=cross_val_score(clf, X, y, cv=5, verbose=1)
    record = clf.predict(testX)

    fina_score = f1_score(testy,record,average='micro')
    
    time_end=time.time()
    
    print("scores = %.5f"%(fina_score))
    # clf = load_clf("dump/svm_emotion.pkl")
    # print("训练完成")
    print("测试完成 , 用时 %.5f 秒"%(time_end - time_start))

if __name__ == "__main__":
    main()
