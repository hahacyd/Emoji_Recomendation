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
    # labelfile = open('model/ingredients/train.solution')
    print("加载数据完成")

    # 序列化
    tfidf_vec = TfidfVectorizer()
    X = tfidf_vec.fit_transform(file)

    # joblib.dump(tfidf_vec, "model/dump/tfidf_vec.vec")
    # joblib.dump(X, "model/dump/X.data")
    # X = joblib.load("model/dump/X.data")

    y = joblib.load("model/dump/y.data")
    print("数据预处理完成")
    
    file.close()
    # labelfile.close()

    # clf = fit(X, y)
    # clf = MultinomialNB()

    print("开始训练")
    time_start = time.time()

    # clf = SGDClassifier(alpha=1e-3,random_state=42, n_jobs=2,max_iter=5)
    # clf = MultinomialNB()
    # clf = LinearSVC(verbose=1)
    clf = DecisionTreeClassifier()
    
    # clf.fit(X, y)
    # joblib.dump(clf,"model/dump/svm_emotion.pkl")
    
    scores=cross_val_score(clf, X, y, cv=5, verbose=1)
    
    time_end=time.time()
    
    print(scores)
    # clf = load_clf("model/dump/svm_emotion.pkl")
    # print("训练完成")
    print("测试完成 , 用时 %.5f 秒"%(time_end - time_start))
    # text_clf = Pipeline([
    #     ('tfidf', TfidfVectorizer(stop_words=stopwords)),
    #     ('clf',MultinomialNB())
    # ])
    # text_clf.fit(file,y)
    # clf = MultinomialNB()
    # while True:
    #     x = input()
    #     data = tfidf_vec.transform([x])
    #     y = clf.predict(data.toarray())
    #     # text_clf.predict([x])
    #     print(le.inverse_transform(y))

if __name__ == "__main__":
    main()
