from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def train():
    X, y = prePrepare()
    print("数据初始化完成")

    trainX,testX,trainy,testy = train_test_split(X,y)

    print("开始训练...",end='')
    clf = MLPClassifier(verbose=1,hidden_layer_sizes=(200,))
    clf.fit(trainX,trainy)

    print("训练完成")
    record = clf.predict(testX)
    fina_score = f1_score(testy,record,average='micro')
    print("score = %.5f"%(fina_score))
    joblib.dump(clf,"dump/mlp.model")
def prePrepare():
    train = np.load("dump/X.npy")
    train = np.array(train)
    # test = np.load("dump/Test.npy")
    y = joblib.load("dump/y.data")
    y = np.array(y)
    print("数据加载成功")
    print(train.shape,y.shape)
    scaler = StandardScaler()
    X_new = scaler.fit_transform(train, y)
    # np.save("dump/Xmlp", X_new)
    joblib.dump(scaler,"dump/scaler")
    return X_new,y
def main():
    train()

if __name__ == "__main__":
    main()