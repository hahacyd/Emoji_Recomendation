from sklearn.externals import joblib
from labelencode import KaggleLabelEncode
from gensim.models import word2vec
def dump_label():
    labelfile = open('model/ingredients/train.solution')
    sentence = labelfile.read()
    labellist = sentence.splitlines()
    # 将 solutions 中标签外的 {} 去除
    labellist = [x.strip("{}") for x in labellist]
    le = joblib.load("model/dump/le.le")
    y = le.transform(labellist)
    joblib.dump(y, "model/dump/y.data")
def dump_word2vec_model():
    file = open("model/train.csv")
    sentence = word2vec.LineSentence(file)

    model = word2vec.Word2Vec(sentences=sentence, size=64)
    model.save("model/dump/word2vec_64d.model")
    file.close()
def dumpCnnTrainData():
    
if __name__ == "__main__":
    dump_word2vec_model()