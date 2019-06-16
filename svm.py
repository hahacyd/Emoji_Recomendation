from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

from labelencode import KaggleLabelEncode
import time

def train():
    file = open("train.csv")
    print("加载数据完成")
    X = joblib.load("dump/X.data")
    
    print(X.shape)

    y = joblib.load("dump/y.data")
    print("数据预处理完成")
    
    file.close()
    print("开始训练")
    time_start = time.time()

    # clf = SGDClassifier(alpha=1e-3,random_state=42, n_jobs=2,max_iter=5)
    # clf = MultinomialNB()
    clf = OneVsOneClassifier(LinearSVC(verbose=1,multi_class='ovr'))
    # clf = DecisionTreeClassifier()
     
    # clf.fit(X, y)
    # joblib.dump(clf,"dump/svm_emotion.pkl")
    scores=cross_val_score(clf, X, y, cv=5, verbose=1)
    
    time_end=time.time()
    
    print(scores)
    # clf = load_clf("dump/svm_emotion.pkl")
    # print("训练完成")
    print("测试完成 , 用时 %.5f 秒" % (time_end - time_start))
if __name__ == "__main__":
    train()