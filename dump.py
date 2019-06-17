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
    # 训练编码器
    le = KaggleLabelEncode()
    le.loaddict("ingredients/emoji.data")

    labelfile = open('ingredients/train.solution')
    sentence = labelfile.read()
    labellist = sentence.splitlines()
    # 将 solutions 中标签外的 {} 去除
    labellist = [x.strip("{}") for x in labellist]

    y = le.transform(labellist)
    # 将已编码的 类别标签保存 以备使用
    np.save("dump/y.npy",np.array(y))
def dump_word2vec_model(size):
    # train_file = open("train.csv")
    # test_file = open("test.csv")
    corpus = open("corpus.csv")
    sentence = word2vec.LineSentence(corpus)
    print("size = %d 预处理完毕 开始训练..."%(size))

    model = word2vec.Word2Vec(sentences=sentence,size = size,window=5)

    print("训练结束")
    # model.save("dump/word2vec_" + str(size) + "d.model")
    model.wv.save("dump/word2vec_" + str(size) + "d.kv")

    corpus.close()
def dump_Cnn_data(size):
    print("size = %d"%(size))
    trainfile = open("train.csv")
    testfile = open("test.csv")

    kv = KeyedVectors.load("dump/word2vec_" + str(size) + "d.kv", mmap='r')
    
    train = transform_to_matrix(kv, trainfile, padding_size=24)
    test = transform_to_matrix(kv, testfile, padding_size=24)

    np.save("dump/Xcnn" + str(size) + ".npy",train)
    np.save("dump/Testcnn" + str(size) + ".npy", test)
    
    trainfile.close()
    testfile.close()
def dump_tfidf_data():
    tfidf_vec = joblib.load("dump/tfidf_vec.vec")
    
    trainfile = open("train.csv")
    testfile = open("test.csv")

    y = joblib.load("dump/y.data")

    train = tfidf_vec.transform(trainfile)
    test = tfidf_vec.transform(testfile)

    # 用 卡方分布 选择特征
    # print("开始用卡方分布筛选特征...",end='')
    # model = SelectKBest(chi2, k=20000)
    # model.fit(train, y)
    
    # print("完成")
    # joblib.dump(model,"dump/feature_selected_20000.chi")
    model = joblib.load("dump/feature_selected_20000.chi")

    trainfile.close()
    testfile.close()

    train = model.transform(train)
    test = model.transform(test)
    np.save("dump/X.npy", train)
    np.save("dump/Test.npy", test)


def dump_tfidf_vec(outpath):
    # trainfile = open("train.csv")
    # testfile = open("test.csv")
    corpus = open("corpus.csv")
    print("数据初始化完毕,训练中 ")
    tfidf_vec = TfidfVectorizer()

    # tfidf_vec.fit(corpus)
    # 所有可获得数据的 tfidf 向量,用于chi2特征选择
    tfidf_vec.fit(corpus)

    print("训练完成")

    joblib.dump(tfidf_vec, outpath)
    corpus.close()
    
    trainfile = open("train.csv")
    testfile = open("test.csv")

    y = joblib.load("dump/y.data")

    train = tfidf_vec.transform(trainfile)
    test = tfidf_vec.transform(testfile)

    # 用 卡方分布 选择特征
    # print("开始用卡方分布筛选特征...",end='')
    # model = SelectKBest(chi2, k=20000)
    # model.fit(train, y)
    
    # print("完成")
    # joblib.dump(model,"dump/feature_selected_20000.chi")
    model = joblib.load("dump/feature_selected_20000.chi")

    trainfile.close()
    testfile.close()

    np.save("dump/X.npy", model.transform(train))
    np.save("dump/Test.npy", model.transform(test))
    
if __name__ == "__main__":
    # dump_word2vec_model(size=128)
    # dump_Cnn_data(size=128)


    # dump_tfidf_vec("dump/tfidf_vec.vec")
    # dump_tfidf_data()
    # dump_tfidf_data("test.csv","dump/Test.data")
    dump_label()