from sklearn.externals import joblib
import numpy as np
class KaggleLabelEncode(object):
    """
    对 kaggle 做的一个labelEncoder

    transform 和 inverse_transform 与sklearn 提供的labelEncoder用法相同
    """
    def __init__(self):
        self.index2value = {}
        self.value2index = {}
        # self.loaddict('model/emoji.data')
    def loaddict(self, path):
        file = open(path, mode='r')
        for line in file:
            sp = line.split()
            self.index2value[int(sp[0])] = sp[1]
        self.value2index = {v:k for k,v in self.index2value.items()}
        # self.value2index = dict(zip(self.index2value.values(),self.index2value.keys()))
        file.close()
    def transform(self, value):
        result = []
        if type(value) == list or type(value) == np.ndarray:
            length = len(value)
            result = [self.value2index[x] for x in value]
            assert(length == len(result))
        elif type(value) == str:
            result = self.value2index[value]
        return result

    def inverse_transform(self, index):
        result = []
        if type(index) == list or type(index) == np.ndarray:
            length = len(index)
            result = [self.index2value[x] for x in index]
            assert(length == len(result))
        elif type(index) == int:
            result = self.index2value[index]
        return result

def main():
    le = KaggleLabelEncode()
    le.loaddict("model/ingredients/emoji.data")
    # print(le.transform(['祈祷','笑']))
    # print(le.inverse_transform([1,46]))
    joblib.dump(le, 'model/dump/le.le')
if __name__ == "__main__":
    main()