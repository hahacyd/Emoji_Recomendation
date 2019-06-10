from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(['hello world', 'hello cyd','haha bob love you'])
# print(X.toarray())
def load_clf(path):
    clf = joblib.load(path)
    return clf

def main():
    clf = load_clf("model/dump/svm_emotion.pkl")
    print("模型加载完毕")
    
    transformer = joblib.load("model/dump/transfer.trans")
    X = joblib.load("model/dump/X.data")
    
    labelfile = open('model/train.solution')
    sentence = labelfile.read()
    labellist = sentence.splitlines()
    # le = LabelEncoder()
    le = joblib.load("model/dump/le.le")
    
    # y = le.fit_transform(labellist)
    y = le.transform(labellist)

    print("数据加载完毕")
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.1)
    
    ypred = clf.predict(testX)
    print(np.mean(ypred == testy))

if __name__ == "__main__":
    main()