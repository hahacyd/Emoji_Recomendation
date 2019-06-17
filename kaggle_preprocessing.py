from sklearn.externals import joblib
from gensim.models import KeyedVectors
import numpy as np
def getTrain_Tfidf_Data():
    """
    返回 tfidf 特征化的训练数据
    
    Return:
    -------
    一个表示 训练数据 tf-idf 特征的稀疏矩阵
    """ 
    X = joblib.load("dump/X.data")
    return X
def getTarget():
    """ 
    返回 训练数据 对应的 label,这些 label 已经根据emoji.data中的对应关系编码

    Return:
    -------
    np.ndarray ,shape = (n_samples,)
    """ 
    return np.load("dump/y.npy")
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
    kv = KeyedVectors.load("dump/word2vec_" + str(size) + "d.kv",mmap='r')
    print("词表加载成功，维度 %d * %d"%(len(kv.vocab) ,kv.vector_size))
    return kv