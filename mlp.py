import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from kaggle_preprocessing import getTarget


def train():
    X, y = prePrepare()

    print("数据初始化完成")

    trainX, testX, trainy, testy = train_test_split(X, y)

    print("开始训练...", end='')
    clf = MLPClassifier(verbose=1, hidden_layer_sizes=(
        400,), warm_start=True, max_iter=1, random_state=1)
    for epoch in range(9):
        clf.fit(trainX, trainy)
        record = clf.predict(testX[0:2])
        fina_score = f1_score(testy[0:2], record, average='micro')
        print("score = %.5f" % (fina_score))
    print("训练完成")

    joblib.dump(clf, "dump/mlp.model")


def KNN():
    X, y = prePrepare()

    print("数据初始化完成")

    trainX, testX, trainy, testy = train_test_split(X, y)

    print("开始训练...", end='')

    clf = KNeighborsClassifier(n_jobs=4)
    clf.fit(trainX, trainy)

    print("训练完成")
    record = clf.predict(testX)
    fina_score = f1_score(testy, record, average='micro')
    print("score = %.5f" % (fina_score))
    # joblib.dump(clf, "dump/mlp.model")


def prePrepare():
    train = np.load("dump/Xmlp64.npy")
    train = np.array(train)
    # test = np.load("dump/Test.npy")
    y = getTarget()
    print("数据加载成功")
    print(train.shape, y.shape)
    scaler = StandardScaler()
    X_new = scaler.fit_transform(train, y)
    # np.save("dump/Xmlp", X_new)
    joblib.dump(scaler, "dump/scaler")
    return X_new, y


def main():
    # train()
    KNN()


if __name__ == "__main__":
    main()
