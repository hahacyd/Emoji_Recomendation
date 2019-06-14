import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

def genTrain():
    file = open("model/ingredients/train.data")
    lines = file.readlines()

    file.close()

    # X
    lines = np.array(lines)
    # y
    y = joblib.load("model/dump/y.data")
    y = np.array(y)
    y = y.astype(np.int32)

    trainX, devX, trainy, devy = train_test_split(X, y, test_size=0.2, random_state=42)

    # 写入训练集 
    # trainfile = open("model/bert/train.tsv",model='x')
    # for label, data in zip(trainy, trainX):
    #     trainfile.write(str(label) + "\t" + data + "\n")
    # trainfile.close()

    # 写入验证集
    devfile = open("model/bert/dev.tsv", model='x')
    for label, data in zip(devy, devX):
        devfile.write(str(label) + "\t" + data + "\n")    
    devfile.close()

    
def main():
    genTrain()

if __name__ == "__main__":
    main()