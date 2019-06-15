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

    trainX, devX, trainy, devy = train_test_split(lines, y, test_size=0.2, random_state=42)

    # 写入训练集 
    trainfile = open("model/bert/train.tsv", mode='x')
    writetext = ''
    for label, data in zip(trainy, trainX):
        writetext += str(label) + "\t" + data
    trainfile.write(writetext) 
    trainfile.close()

    # 写入验证集
    devfile = open("model/bert/dev.tsv", mode='x')

    writetext = ''
    for label, data in zip(devy, devX):
        writetext += str(label) + "\t" + data
    devfile.write(writetext)    
    devfile.close()

def genTest():
    input_filename = "model/ingredients/test.data"
    output_filename = "model/bert/test.tsv"
    file = open(input_filename)
    testfile = open(output_filename,mode='x')
    sentence = ''
    # sentence = file.read()
    for line in file:
        sentence += line.split(maxsplit=1)[1]
    file.write(sentence)
    file.close()
    print("完成")
    
def main():
    # genTrain()
    genTest()

if __name__ == "__main__":
    main()