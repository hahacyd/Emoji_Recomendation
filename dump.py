from sklearn.externals import joblib
# from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec,KeyedVectors
from word2vec_lac import sentence2index,sentence2vecter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2
import numpy as np
class KaggleLabelEncode(object):
    """
    对 kaggle 做的一个labelEncoder

    transform 和 inverse_transform 与sklearn 提供的labelEncoder用法相同
    """
    def __init__(self):
        self.index2value = {}
        self.value2index = {}
        # self.loaddict('emoji.data')
    def loaddict(self, path):
        """ 
        从文件 中获得 编码方式，这里仅仅指 emoji.data 文件中的编码方式
        """ 
        file = open(path, mode='r')
        for line in file:
            sp = line.split()
            self.index2value[int(sp[0])] = sp[1]
        self.value2index = {v:k for k,v in self.index2value.items()}
        # self.value2index = dict(zip(self.index2value.values(),self.index2value.keys()))
        file.close()
    def transform(self, value):
        """ 
        将 字符 转换为 编码，实际上就是emoji.data中 表情所对应的数字
        """
        if self.dict_check() == False:
            print("请先执行 loaddict 来加载编码方式")
        result = []
        if type(value) == list or type(value) == np.ndarray:
            length = len(value)
            result = [self.value2index[x] for x in value]
            assert(length == len(result))
        elif type(value) == str:
            result = self.value2index[value]
        return result

    def inverse_transform(self, index):
        """
        将编码 还原为数字,transform 的逆操作 
        """
        if self.dict_check() == False:
            print("请先执行 loaddict 来加载编码方式")
        result = []
        if type(index) == list or type(index) == np.ndarray:
            length = len(index)
            result = [self.index2value[x] for x in index]
            assert(length == len(result))
        elif type(index) == int:
            result = self.index2value[index]
        return result
    def dict_check(self):
        """
        检查这个编码器是否能工作，通过检查loaddict是否已被运行
        """
        return len(self.index2value) > 0
def dump_label():
    """ 
    将训练数据编码，并把结果 dump 到磁盘上
    """
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
    np.save("dump/y.npy", np.array(y))
    
    return np.array(y)
def dump_word2vec_model(size):
    # train_file = open("train.csv")
    # test_file = open("test.csv")
    corpus = open("corpus.csv")
    sentence = word2vec.LineSentence(corpus)
    print("size = %d 预处理完毕 开始训练..."%(size))

    model = word2vec.Word2Vec(sentences=sentence,size = size,min_count=0)

    print("训练结束")
    # model.save("dump/word2vec_" + str(size) + "d.model")
    model.wv.save("dump/word2vec_" + str(size) + "d.kv")

    corpus.close()
    return model.wv
def dump_Cnn_data(size):
    """ 
    生成 Cnn 所需要的训练和测试数据，并将这些数据 dump 到磁盘上

    Parameter :
    -----------
    size : 确定词向量的维数

    Return :
    --------
    np.ndarray, shape = (n_samples,size)
    """
    print("size = %d"%(size))
    trainfile = open("train.csv")
    testfile = open("test.csv")

    kv = KeyedVectors.load("dump/word2vec_" + str(size) + "d.kv", mmap='r')
    
    train = sentence2index(kv, trainfile, padding_size=24)
    test = sentence2index(kv, testfile, padding_size=24)

    np.save("dump/Xcnn" + str(size) + ".npy",train)
    np.save("dump/Testcnn" + str(size) + ".npy", test)
    
    trainfile.close()
    testfile.close()
    return train,test
def dump_Mlp_data(size):
    print("size = %d"%(size))
    trainfile = open("train.csv")
    testfile = open("test.csv")
    kv = KeyedVectors.load("dump/word2vec_" + str(size) + "d.kv", mmap='r')
    
    train = sentence2vecter(kv, trainfile)
    test = sentence2vecter(kv, testfile)
    
    np.save("dump/Xmlp" + str(size) + ".npy",train)
    np.save("dump/Testmlp" + str(size) + ".npy", test)

    trainfile.close()
    testfile.close()
    return train,test
def dump_tfidf_data():
    corpus = open("corpus.csv")
    print("数据初始化完毕,训练中 ")
    tfidf_vec = TfidfVectorizer()

    # 所有可获得数据的 tfidf 向量,用于chi2特征选择
    tfidf_vec.fit(corpus)

    print("tfidf 特征提取模型 训练完成")
    # tfidf_vec = joblib.load("dump/tfidf_vec.vec")
    
    trainfile = open("train.csv")
    testfile = open("test.csv")

    # y = getTarget()
    y = np.load("dump/y.npy")

    train = tfidf_vec.transform(trainfile)
    test = tfidf_vec.transform(testfile)

    # 用 卡方分布 选择特征
    print("开始用卡方分布筛选特征...",end='')
    model = SelectKBest(chi2, k=20000)
    model.fit(train, y)
    
    print("完成")
    # joblib.dump(model,"dump/feature_selected_20000.chi")
    # model = joblib.load("dump/feature_selected_20000.chi")

    trainfile.close()
    testfile.close()

    train = model.transform(train)
    test = model.transform(test)
    np.save("dump/X.npy", train)
    np.save("dump/Test.npy", test)

    return train,test
if __name__ == "__main__":
    # dump_word2vec_model(size=64)
    # dump_Cnn_data(size=64)

    # dump_tfidf_data()
    # dump_tfidf_data("test.csv","dump/Test.data")
    # dump_label()
    dump_Mlp_data(64)