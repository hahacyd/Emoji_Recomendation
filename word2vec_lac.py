from gensim.models import word2vec
from gensim.models import KeyedVectors
import numpy as np
from sklearn.externals import joblib
def transform_to_matrix(kv, x, padding_size):
    """
    将数据集 x 中的每个sentence中的词语替换为其在 word2vec 模型中的index
    
    作为pytorch中的卷积神经网络的输入数据

    Parameter:
    ----------
    kv : word2vec模型

    x : 数据集，一般包含多个sentences

    padding_size : 默认一个sentence包含多少个词语，若某个sentence长度比padding_size短，则补0,否则截断
    
    Return:
    -------
    np.ndarray 类型，shape = (n_samples,n_features)
    """
    lines = x.readlines()
    line_num = len(lines)
    res = np.zeros((line_num, padding_size))
    sentence = []
    print("开始转换...",end='')
    for i in range(len(lines)):
        print("\r%d/%d" % (i, line_num), end='')
        sentence.clear()
        sentence.extend(lines[i].split())
        for j in range(len(sentence)):
            # 每个句子 只考虑前 padding_size 个词
            if j >= padding_size:
                break 
            try:
                res[i, j] = kv.vocab[sentence[j]].index
            except:
                print("模型中没有词语 %s, 将自动补0"%(sentence[j]))
    return res    
def getCnnTrainData():
    file = open("train.csv")
    model = word2vec.Word2Vec.load("dump/word2vec_32d.model")
    inputs = transform_to_matrix(model, file, padding_size=12)
    # joblib.dump(inputs, "dump/Xcnn.csv")
    np.save("dump/Xcnn.npy",inputs)
    file.close()
    return inputs

def main():
    getCnnTrainData()
if __name__ == "__main__":
    main()