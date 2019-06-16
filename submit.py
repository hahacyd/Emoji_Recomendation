from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.externals import joblib
import torch
import torch.utils.data as Data
from cnn import SentimentCNN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_clf(path):
    clf = joblib.load(path)
    return clf
def NaiveBayes_clf():
    clf = joblib.load("dump/naive_bayes.model")
    test = joblib.load("dump/Test.data")

    return clf.predict(test)

def svm_clf():
    clf = load_clf("dump/svm_emotion.pkl")
    print("训练模型加载完毕")
    
    tfidf_vec = joblib.load("dump/tfidf_vec.vec")

    test_file = open("test.csv")
    testX = tfidf_vec.transform(test_file)
    print("测试数据加载完毕")

    ypred = clf.predict(testX)
    return ypred

def cnn_clf():
    BATCH_SIZE = 100
    model = torch.load("dump/net_345_128_adam_1.model",map_location=device)
    model.eval()

    testX = np.load("dump/Testcnn128.npy")
    length = len(testX)

    dataset = Data.TensorDataset(torch.from_numpy(testX).long())
    data_loader = Data.DataLoader(dataset, batch_size=BATCH_SIZE)

    intervel = 100
    record = np.empty(0,dtype=np.int32)
    for batch_index, inputs in enumerate(data_loader, 0):
        inputs = inputs[0]
        inputs = inputs.to(device)
        ypred = model(inputs)
        ypred = ypred.detach().cpu().numpy()
        ypred = ypred.argmax(axis=1)
        record = np.hstack([record,ypred])
        if batch_index != 0 and batch_index % intervel == 0:
            print("[ %d/%d ]" % (batch_index / intervel, len(dataset) / (BATCH_SIZE * intervel)))
            
    # ypred = model(torch.from_numpy(testX).long().to(device))
    # ypred = model(torch.from_numpy(testX).long())
    assert(length == len(record))
    print("预测完毕")
    return record
def main():

    ypred = cnn_clf()
    # ypred = NaiveBayes_clf()
    # 开始将结果 写入提交文件
    submit_file = open("submit/submit.csv", mode='w')
    submit_file.writelines(["ID,Expected\n"])
    text = ''
    for index, value in enumerate(ypred,0):
        text += str(index) + ',' + str(value) + '\n'
    submit_file.write(text)

    submit_file.close()
    # print(np.mean(ypred == testy))

if __name__ == "__main__":
    main()