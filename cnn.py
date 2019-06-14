import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score

from labelencode import KaggleLabelEncode
from gensim.models import word2vec
from word2vec_lac import getCnnTrainData
import time
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("gpu")
else:
    print("cpu")
# device = torch.device("cpu")
def batch_data(X,batch_size = 100):
    length = X.shape[0]
    for_times = math.floor(length / 100)

    for i in range(for_times):
        yield X[i * batch_size:(i + 1) * batch_size]
    yield X[for_times * batch_size :]
def getEmbed_lookup(size):
    # model = word2vec.Word2Vec.load("model/dump/word2vec_32d.model")
    model = word2vec.Word2Vec.load("model/dump/word2vec_" + str(size) + "d.model")
    print("词表加载成功，维度 %d * %d"%(len(model.wv.vocab) ,model.wv.vector_size))
    return model.wv
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords
def kaggle_preprocessing(size):
    X = np.load("model/dump/Xcnn"+str(size) + ".npy")
    y = joblib.load("model/dump/y.data")
    y = np.array(y)
    y = y.astype(np.int32)
    print("数据预处理完成")
    return X,y
class SentimentCNN(nn.Module):
    """
    The embedding layer + CNN model that will be used to perform sentiment analysis.
    """
    def __init__(self, embed_model, vocab_size, output_size, embedding_dim,
                 num_filters=100, kernel_sizes=[3, 4, 5], freeze_embeddings=True, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentCNN, self).__init__()

        # set class vars
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim
        
        # 1. embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # set weights to pre-trained
        self.embedding.weight = nn.Parameter(torch.from_numpy(embed_model.vectors)) # all vectors
        # (optional) freeze embedding weights
        if freeze_embeddings:
            self.embedding.requires_grad = False
        
        # 2. convolutional layers
        self.conv2d = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim), padding=(k-2,0)) 
            for k in kernel_sizes])
        
        # 3. final, fully-connected layer for classification
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_size) 
        
        # 4. dropout and sigmoid layers
        self.dropout = nn.Dropout(drop_prob)
        # self.softmax = nn.Softmax()
    def conv_and_pool(self, x, conv):
        """
        Convolutional + max pooling layer
        """
        # squeeze last dim to get size: (batch_size, num_filters, conv_seq_length)
        # conv_seq_length will be ~ 200
        x = conv(x)
        x = F.relu(x)
        x = x.squeeze(3)
        
        # 1D pool over conv_seq_length
        # squeeze to get size: (batch_size, num_filters)
        x_max = F.max_pool1d(x, x.size(2))
        x_max = x_max.squeeze(2)
        return x_max

    def forward(self, x):
        """
        Defines how a batch of inputs, x, passes through the model layers.
        Returns a single, sigmoid-activated class score as output.
        """
        # embedded vectors
        # x = torch.LongTensor(x)
        embeds = self.embedding(x) # (batch_size, seq_length, embedding_dim)
        # embeds.unsqueeze(1) creates a channel dimension that conv layers expect
        embeds = embeds.unsqueeze(1)
        
        # get output of each conv-pool layer
        conv_results = [self.conv_and_pool(embeds, conv) for conv in self.conv2d]
        
        # concatenate results and add dropout
        x = torch.cat(conv_results, 1)
        x = self.dropout(x)
        
        # 这里是一个测试 100 * 100 * 2
        # x = torch.reshape(x,(x.size(0),x.size(1) * x.size(2)))
        # final logit
        logit = self.fc(x) 
        
        # sigmoid-activated --> a class score
        # return self.softmax(logit)
        return logit

vector_size = 64
embed_lookup = getEmbed_lookup(size=vector_size)

vocab_size = len(embed_lookup.vocab)
output_size = 72
embedding_dim = embed_lookup.vector_size
num_filters = 100
kernel_sizes = [3,4,5]

BATCH_SIZE=200
X, y = kaggle_preprocessing(size=vector_size)
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=42)
net = SentimentCNN(embed_lookup,vocab_size,output_size,embedding_dim,kernel_sizes=kernel_sizes)

net = net.to(device)
def train_CNN():

    # print(net)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters())

    net.train()
 
    running_loss = 0.0  

    for epoch in range(2):
        dataset = Data.TensorDataset(torch.from_numpy(trainX).long(),torch.from_numpy(trainy).long())
        data_loader = Data.DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)

        for batch_index, (inputs, labels) in enumerate(data_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs,labels)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            intervel = 100
            if batch_index != 0 and batch_index % 100 == 0:
                print("epoch = %d [ %d/%d ] loss = %f" % (epoch, batch_index / intervel, len(dataset) / (intervel * BATCH_SIZE), running_loss/BATCH_SIZE))    
                running_loss = 0.0
        test_CNN()
    torch.save(net,"model/dump/net_345_64_adam.model")

def test_CNN():
    model = net
    # model = torch.load("model/dump/net_345_adam.model",map_location=device)
    model.eval()

    dataset = Data.TensorDataset(torch.from_numpy(testX).long(),torch.from_numpy(testy))
    data_loader = Data.DataLoader(dataset, batch_size=BATCH_SIZE)
    score = 0.0
    record = np.empty(0)
    for batch_index, (inputs, labels) in enumerate(data_loader, 0):
        # print(np.sum(labels.numpy() == testy[batch_index * BATCH_SIZE:(batch_index + 1)*BATCH_SIZE]))

        inputs = inputs.to(device)
        
        ypred = model(inputs)
        ypred = ypred.cpu().detach().numpy()

        ypred = ypred.argmax(axis = 1)

        record = np.hstack([record,ypred])

        labels = labels.numpy()

        score += f1_score(labels,ypred,average='micro')
        intervel = 100
        if batch_index != 0 and batch_index % intervel == 0:
            print("[ %d/%d ] score = %f" % (batch_index / intervel, len(dataset) / (intervel * BATCH_SIZE), score / BATCH_SIZE))    
            score = 0
    print("总f1_score = %.5f"%(f1_score(testy,record,average='micro')))
def main():
    print("开始训练")
    start_time = time.time()

    train_CNN()

    end_time = time.time()
    print("训练完成,用时 %.5f mins"%((end_time - start_time)/60))

    # test_CNN()
if __name__ == "__main__":
    main()
    # for x in batch_data(testX):
    #     print(x.shape)
