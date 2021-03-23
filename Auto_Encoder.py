import torch
import torch.nn as nn
import torch.utils.data as Data
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),  # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
     transforms.Normalize((0.5, 0.5), (0.5, 0.5))])


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_root):
        self.data = data_root
        # self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        # labels = self.label[index]
        return data  # , labels

    def __len__(self):
        return len(self.data)


# 超参数
# DATA_DIM = 10
EPOCH = 10
BATCH_SIZE = 64
LR = 0.001
BIAS = 0.005
# SAMPLE_SIZE = 10
NAME = "1002_002803.csv"
'''
read csv file to numpy form
'''


def read_csv_file_data(file_path):
    data = pd.read_csv(file_path)
    data = data.iloc[:, 1:]
    columns = data.columns.tolist()
    newDataFrame = pd.DataFrame(index=data.index)
    for c in columns:
        d = data[c]
        newDataFrame[c] = d.tolist()
    train_data = np.array(newDataFrame, dtype=np.float32)
    train_x_list = torch.from_numpy(train_data)
    return data, train_x_list


'''
read txt file to numpy form
'''


def read_txt_file_data(filepath):
    data = list()
    for line in open(filepath, 'r'):
        temp = torch.zeros(784)
        tt = line.split(' ')[:-1]
        # tt = tt[:-1]

        for item in tt:
            content = item.split(':')
            temp[int(content[0])] = float(content[1])
        data.append(temp)
    return data[10:len(data) - 10]


file_path = "D:/dataset/norm_data/" + NAME
raw_file_path = "D:/dataset/raw_data/" + NAME
rawdata, traindata = read_csv_file_data(file_path)
raw_data = pd.read_csv(raw_file_path)
max_price = raw_data.iloc[:, 2].max()
min_price = raw_data.iloc[:, 2].min()
DATA_DIM = traindata.shape[1]
HIDE_DIM = 20
train_data = MyDataset(traindata)
trainLoader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=False)


class AutoEncoder(nn.Module):
    def __init__(self, _input_dim, _output_dim):
        super(AutoEncoder, self).__init__()
        self.input_dim = _input_dim
        self.out_dim = _output_dim
        # 压缩
        self.dropout = nn.Dropout(0.2)
        self.encoder = nn.Sequential(
            nn.Linear(_input_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, 32),
            nn.Sigmoid(),
            nn.Linear(32, 8),
            nn.Sigmoid(),
            nn.Linear(8, _output_dim),  # 压缩成3个特征, 进行 3D 图像可视化
        )
        # 解压
        self.decoder = nn.Sequential(
            nn.Linear(_output_dim, 8),
            nn.Sigmoid(),
            nn.Linear(8, 32),
            nn.Sigmoid(),
            nn.Linear(32, 128),
            nn.Sigmoid(),
            nn.Linear(128, _input_dim)
        )

    def forward(self, x):
        x = self.dropout(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def predict(self, x):
        return self.encoder(x).detach()


def read_mnist_data():
    f = open('D:/Java_workspace/handwrite/data2.txt')
    i = 0
    data = []
    for line in f:
        temp = torch.zeros(784)
        tt = line.split(' ')
        tt = tt[1:]
        for item in tt:
            content = item.split(':')
            temp[int(content[0])] = float(content[1])
        data.append(temp)
    return data


def draw_mnist(data, title="raw data"):
    data = np.array(data)
    img = data.reshape(28, 28)
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()


def draw_data(data, title="raw data"):
    data = np.array(data)
    plt.title(title)
    plt.plot(data)
    plt.show()


autoencoder = AutoEncoder(DATA_DIM, HIDE_DIM)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()


def learn():
    loss_temp = []
    epoch = 0
    autoencoder.train()
    count = 0
    while count < 5:
        epoch += 1
        loss_sum = 0
        step = 0
        for _, (x) in enumerate(trainLoader):
            step += 1
            x = torch.tensor(x)
            b_x = x.view(-1, DATA_DIM)
            b_y = x.view(-1, DATA_DIM)
            encoded, decoded = autoencoder(b_x)
            loss = loss_func(decoded, b_y)
            loss_sum += loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss <= BIAS:
                count += 1
            else:
                count = 0
        if epoch % 100 == 0:
            print('epoch: ' + str(epoch) + ' loss : ' + str(loss.data.item()))
    print('epoch:= ' + str(epoch))
    # torch.save(autoencoder, "model/autoencoder_"+file_path+".pt")


def predict():
    autoencoder.eval()
    result = []
    row = []
    res = []
    for item in range(len(traindata)):
        x = traindata[item].unsqueeze(0)
        encoder_out, decoder_out = autoencoder(x)
        predict_encoder = autoencoder.predict(x)
        predict_encoder = torch.squeeze(predict_encoder)
        decoder_out = decoder_out.detach()
        decoder_out = torch.squeeze(decoder_out)
        res.append(predict_encoder.numpy())
        result.append(decoder_out.numpy()[1] * (max_price - min_price) + min_price)
        row.append(traindata[item][1] * (max_price - min_price) + min_price)

    file_path2 = "D:/dataset/new_processed/" + NAME
    result_1 = pd.DataFrame(data=res)
    result_1.to_csv(file_path2, index=False)
    draw_data(row)
    draw_data(result, "out_put data")


learn()
# autoencoder = torch.load("model/autoencoder.pt")

predict()
