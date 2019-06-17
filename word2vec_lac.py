from gensim.models import word2vec
from gensim.models import KeyedVectors
import numpy as np
from sklearn.externals import joblib
def sentence2vecter(kv, x):
    """ 
    将 sentence 转成一个词向量，这个词向量产生于将 sentence 中所有词的词向量相加，然后平均

    Parameters:
    -----------
    kv : word2vec 模型
    x : raw_document 训练数据集, 

    Return:
    -------
    np.ndarray , shape = (n_samples,dimensions), dimensions 是 kv 模型中词向量的维数
    """
    lines = x.readlines()
    line_num = len(lines)

    vector_size = kv.vector_size
    result = np.zeros((line_num, vector_size))

    sentence = []
    temp = np.zeros(vector_size)
    print("开始转换...", end = '')
    ignore_count = 0
    for i,line in enumerate(lines,0):
        print("\r%d/%d" % (i, line_num), end='')
        
        sentence.clear()
        sentence.extend(line.split())
        temp *= 0
        for word in sentence:
            try:
                temp += kv[word]
            except:
                # print("模型中没有词语 %s"%(word))
                temp += 0
                ignore_count +=1
        # 取词向量之和的平均值
        if len(sentence) != 0 :
            temp = temp / len(sentence)
        result[i] = temp
    print("\n%d 个词被忽略"%(ignore_count))
    return result       
def sentence2index(kv, x, padding_size):
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
    np.ndarray 类型，shape = (n_samples,n_featuresult)
    """
    lines = x.readlines()
    line_num = len(lines)
    result = np.zeros((line_num, padding_size))
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
                result[i, j] = kv.vocab[sentence[j]].index
            except:
                print("模型中没有词语 %s, 将自动补0"%(sentence[j]))
    return result    
def getCnnTrainData():
    file = open("train.csv")
    model = word2vec.Word2Vec.load("dump/word2vec_32d.model")
    inputs = sentence2index(model, file, padding_size=12)
    # joblib.dump(inputs, "dump/Xcnn.csv")
    np.save("dump/Xcnn.npy",inputs)
    file.close()
    return inputs

def main():
    getCnnTrainData()
if __name__ == "__main__":
    main()