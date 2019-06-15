from gensim.models import word2vec
from gensim.models import KeyedVectors
import numpy as np
from sklearn.externals import joblib
def transform_to_matrix(kv, x, padding_size):
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
                v = 1
    return res    
def getCnnTrainData():
    file = open("model/train.csv")
    model = word2vec.Word2Vec.load("model/dump/word2vec_32d.model")
    inputs = transform_to_matrix(model, file, padding_size=12)
    # joblib.dump(inputs, "model/dump/Xcnn.csv")
    np.save("model/dump/Xcnn.npy",inputs)
    file.close()
    return inputs

def main():

    getCnnTrainData()
if __name__ == "__main__":
    main()