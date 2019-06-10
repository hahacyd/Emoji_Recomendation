from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.externals import joblib
def load_clf(path):
    clf = joblib.load(path)
    return clf

def main():
    clf = load_clf("model/dump/svm_emotion.pkl")
    print("训练模型加载完毕")
    
    tfidf_vec = joblib.load("model/dump/tfidf_vec.vec")

    test_file = open("model/test.csv")
    testX = tfidf_vec.transform(test_file)
    print("测试数据加载完毕")

    ypred = clf.predict(testX)

    # 开始将结果 写入提交文件
    submit_file = open("model/submit/submit.csv", mode='w')
    submit_file.writelines(["ID,Expected\n"])
    text = ''
    for index, value in enumerate(ypred,0):
        text += str(index) + ',' + str(value) + '\n'
    submit_file.write(text)

    test_file.close()
    submit_file.close()
    # print(np.mean(ypred == testy))

if __name__ == "__main__":
    main()