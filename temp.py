from sklearn.externals import joblib
from labelencode import KaggleLabelEncode
import numpy as np


x = np.load("model/dump/Xcnn.npy")
x = x.astype(np.int32)
np.save("model/dump/Xcnn.npy",x)