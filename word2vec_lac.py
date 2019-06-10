from gensim.models import word2vec
import numpy as np
from sklearn.externals import joblib
def transform_to_matrix(model, x, padding_size):
    lines = x.readlines()
    line_num = len(lines)
    # vec_size = model.wv.vector_size
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
                res[i, j] = model.wv.vocab[sentence[j]].index
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
def getEmbed_lookup():
    model = word2vec.Word2Vec.load("model/dump/word2vec_32d.model")
    # inputs = transform_to_matrix2(model, file, padding_size=12)
    return model.wv
def main():

    getCnnTrainData()
if __name__ == "__main__":
    main()