import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split

from labelencode import KaggleLabelEncode
from gensim.models import word2vec
from word2vec_lac import getCnnTrainData
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
def getEmbed_lookup():
    model = word2vec.Word2Vec.load("model/dump/word2vec_32d.model")
    # inputs = transform_to_matrix2(model, file, padding_size=12)
    return model.wv
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords
def load_clf(path):
    clf = joblib.load(path)
    return clf
def kaggle_preprocessing():
    # file = open("model/train.csv")
    # X = file.readlines()
    # X = np.array(X)
    X = np.load("model/dump/Xcnn.npy")
    # labelfile = open('model/ingredients/train.solution')
    # file.close()
    # labelfile.close()
    # print("加载数据完成")

    # 序列化
    # tfidf_vec = TfidfVectorizer()
    # X = tfidf_vec.fit_transform(file)
    # # joblib.dump(tfidf_vec, "model/dump/tfidf_vec.vec")
    # # joblib.dump(X, "model/dump/X.data")
    # # X = joblib.load("model/dump/X.data")
    # sentence = labelfile.read()
    # labellist = sentence.splitlines()
    # # 将 solutions 中标签外的 {} 去除
    # labellist = [x.strip("{}") for x in labellist]
    # le = joblib.load("model/dump/le.le")
    # y = le.transform(labellist)
    # X = getCnnTrainData()
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
        self.convs_1d = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim), padding=k-2) 
            for k in kernel_sizes])
        
        # 3. final, fully-connected layer for classification
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_size) 
        
        # 4. dropout and sigmoid layers
        self.dropout = nn.Dropout(drop_prob)
        # self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
    
    def conv_and_pool(self, x, conv):
        """
        Convolutional + max pooling layer
        """
        # squeeze last dim to get size: (batch_size, num_filters, conv_seq_length)
        # conv_seq_length will be ~ 200
        x = F.relu(conv(x))
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
        embeds = self.embedding(x.long()) # (batch_size, seq_length, embedding_dim)
        # embeds.unsqueeze(1) creates a channel dimension that conv layers expect
        embeds = embeds.unsqueeze(1)
        
        # get output of each conv-pool layer
        conv_results = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        
        # concatenate results and add dropout
        x = torch.cat(conv_results, 1)
        x = self.dropout(x)
        
        # final logit
        logit = self.fc(x) 
        
        # sigmoid-activated --> a class score
        return self.softmax(logit)
class Net(nn.Module):
    def __init__(self, embed_model, vocab_size, output_size, embedding_dim,
                num_filters=100,kernel_sizes=[3,4,5],freeze_embeddings = True,drop_prob=0.5):
        super(Net, self).__init__()
        # set class vars
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim

        # 1. embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # set weights to pre-trained
        self.embedding.weight = nn.Parameter(torch.from_numpy(embed_model.vectors))  #all vectors
        
        if freeze_embeddings:
            self.embedding.requires_grad = False

        # 2 convolutional layers
        self.convs_1d = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim), padding=k - 2)
            for k in kernel_sizes
        ])

        # 3 final ,fully-connected layer for classification
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_size)
        
        # 4. dropout and sigmoid layers
        self.dropout = nn.Dropout(drop_prob)
        self.softmax = nn.Softmax()
        # self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # self.fc1 = nn.Linear(4*4*50, 500)
        # self.fc2 = nn.Linear(500, 10)
        # self.pool = nn.MaxPool2d(2,2)
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x_max = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x_max
    def forward(self, x):
        # embedded vectors
        embeds = self.embedding(x)# (batch_size, seq_length, embedding_dim)
        # embeds.unsqueeze(1) creates a channel dimension that conv layers expect
        embeds = embeds.unsqueeze(1)

        # get output of each conv-pool layer
        conv_results = [self.conv_add_pool(embeds,conv) for conv in self.convs_1d]

        # concatenate results and add dropout
        x = torch.cat(conv_results, 1)
        x = self.dropout(x)

        # final logit
        logit = self.fc(x)

        return self.softmax(logit)

        # x = self.pool(F.relu(self.conv1(x)))
        # x = x.view(-1, 4*4*50)
        # x = F.relu(self.fc1(x))
        # x = F.log_softmax(x, dim=1)
        # return x
embed_lookup = getEmbed_lookup()

vocab_size = len(embed_lookup.vocab)
output_size = 72
embedding_dim = embed_lookup.vector_size
num_filters = 100
kernel_sizes = [3,4,5]

# net = Net(embed_lookup,vocab_size,output_size,embedding_dim,num_filters,kernel_sizes)
net = SentimentCNN(embed_lookup,vocab_size,output_size,embedding_dim)

net = net.to(device)
print(net)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=1e-2,momentum=0.9)
BATCH_SIZE=100
X, y = kaggle_preprocessing()
trainX,testX,trainy,testy = train_test_split(X,y,test_size=0.2,random_state=42)
def train_CNN():
    net.train()
 
    running_loss = 0.0  

    dataset = Data.TensorDataset(torch.from_numpy(trainX),torch.from_numpy(trainy))
    data_loader = Data.DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
    
    for batch_index, (inputs, labels) in enumerate(data_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs,labels)
        running_loss = loss.item()

        loss.backward()
        optimizer.step()
        print("[ %d/%d ] loss = %f" % (batch_index, batch_index * BATCH_SIZE / len(dataset), running_loss))
    

def test_CNN():
    pass
def main():
    print("开始训练")
    train_CNN()
    print("训练完成")
    # while True:
    #     x = input()
    #     data = tfidf_vec.transform([x])
    #     y = clf.predict(data.toarray())
    #     # text_clf.predict([x])
    #     print(le.inverse_transform(y))

if __name__ == "__main__":
    main()
