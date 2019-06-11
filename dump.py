from sklearn.externals import joblib
from labelencode import KaggleLabelEncode
from gensim.models import word2vec
from word2vec_lac import transform_to_matrix
import numpy as np
def dump_label():
    labelfile = open('model/ingredients/train.solution')
    sentence = labelfile.read()
    labellist = sentence.splitlines()
    # 将 solutions 中标签外的 {} 去除
    labellist = [x.strip("{}") for x in labellist]
    le = joblib.load("model/dump/le.le")
    y = le.transform(labellist)
    joblib.dump(y, "model/dump/y.data")
def dump_word2vec_model(size):
    # train_file = open("model/train.csv")
    # test_file = open("model/test.csv")
    corpus = open("model/corpus.csv")
    sentence = word2vec.LineSentence(corpus)
    print("size = %d 预处理完毕 开始训练..."%(size))

    model = word2vec.Word2Vec(sentences=sentence, size=size)

    print("训练结束")
    model.save("model/dump/word2vec_" + str(size) + "d.model")

    corpus.close()
def dump_Cnn_data(file, size):
    print("%s : size = %d"%(file,size))
    file = open(file)

    model = word2vec.Word2Vec.load("model/dump/word2vec_" + str(size) + "d.model")
    inputs = transform_to_matrix(model, file, padding_size=24)
    # joblib.dump(inputs, "model/dump/Xcnn.csv")
    np.save("model/dump/Testcnn" + str(size) + ".npy",inputs)
    file.close()
if __name__ == "__main__":
    # dump_word2vec_model(size=32)
    dump_Cnn_data(file = "model/test.csv",size=32)