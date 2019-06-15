from sklearn.externals import joblib
from labelencode import KaggleLabelEncode
# from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec,KeyedVectors
from word2vec_lac import transform_to_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2
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

    model = word2vec.Word2Vec(sentences=sentence,size = size,window=8)

    print("训练结束")
    # model.save("model/dump/word2vec_" + str(size) + "d.model")
    model.wv.save("model/dump/word2vec_" + str(size) + "d.kv")

    corpus.close()
def dump_Cnn_data(file, size):
    print("%s : size = %d"%(file,size))
    file = open(file)

    # model = word2vec.Word2Vec.load("model/dump/word2vec_" + str(size) + "d.model")
    kv = KeyedVectors.load("model/dump/word2vec_" + str(size) + "d.kv",mmap='r')
    inputs = transform_to_matrix(kv, file, padding_size=24)
    # np.save("model/dump/Xcnn" + str(size) + ".npy",inputs)
    np.save("model/dump/Testcnn" + str(size) + ".npy",inputs)
    file.close()
def dump_tfidf_data(inputpath,outputpath):
    file = open(inputpath)
    tfidf_vec = joblib.load("model/dump/tfidf_vec.vec")
    
    res = tfidf_vec.transform(file)
    joblib.dump(res, outputpath)
    
    file.close()


def dump_tfidf_vec(outpath):
    # trainfile = open("model/train.csv")
    # testfile = open("model/test.csv")

    corpus = open("model/corpus.csv")
    print("数据初始化完毕,训练中 ")
    tfidf_vec = TfidfVectorizer()

    tfidf_vec.fit(corpus)
    print("训练完成")
    joblib.dump(tfidf_vec, outpath)
    corpus.close()
if __name__ == "__main__":
    # dump_word2vec_model(size=256)
    # dump_Cnn_data(file = "model/train.csv",size=256)
    dump_Cnn_data(file = "model/test.csv",size=256)


    # dump_tfidf_vec("model/dump/tfidf_vec.vec")
    # dump_tfidf_data("model/train.csv","model/dump/X.data")
    # dump_tfidf_data("model/test.csv","model/dump/Test.data")