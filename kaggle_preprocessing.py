import numpy as np
from gensim.models import KeyedVectors
from sklearn.externals import joblib

from dump import dump_Cnn_data, dump_label,dump_word2vec_model,dump_tfidf_data


def getTrain_Tfidf_Data():
    """
    返回 tfidf 特征化的训练数据

    Return:
    -------
    一个表示 训练数据 tf-idf 特征的稀疏矩阵
    """
    try:
        X = joblib.load("dump/X.data")
    except:
        print("未找到 训练数据，现在开始生成...")
        X ,test = dump_tfidf_data()
    return X


def getTarget():
    """ 
    返回 训练数据 对应的 label,这些 label 已经根据emoji.data中的对应关系编码

    Return:
    -------
    np.ndarray ,shape = (n_samples,)
    """
    try:
        y = np.load("dump/y.npy")
    except:
        print("未找到 标记数据，现在开始生成...")
        y = dump_label()
    return y


def getCnn_Data(vector_size):
    """ 
    返回 Cnn 所需要的训练数据
    """ 
    try:
        X = np.load("dump/Xcnn"+str(vector_size) + ".npy")
    except:
        print("未找到 训练数据，现在开始生成...")
        X, Xtest = dump_Cnn_data(vector_size)
    return X

def getCnn_Test(vector_size):
    """ 
    返回用于 Cnn 模型的测试集数据
    """ 
    try:
        testX = np.load("dump/Testcnn64.npy")
    except:
        print("未找到 训练数据，现在开始生成...")
        X, Xtest = dump_Cnn_data(vector_size)
    return testX

def getEmbed_lookup(size):
    """ 
    返回 文件中 保存的词向量,当相应的模型不存在时，将调用训练函数训练并返回此模型

    Parameter :
    -----------
    size : 词向量的维度

    Return :
    --------
    对应此维度的 word2vec 模型
    """
    try:
        kv = KeyedVectors.load("dump/word2vec_" + str(size) + "d.kv", mmap='r')
    except:
        print("未找到 词向量模型，现在开始生成...")
        kv = dump_word2vec_model(size)
    print("词表加载成功，维度 %d * %d" % (len(kv.vocab), kv.vector_size))
    return kv
