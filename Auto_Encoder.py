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
BIAS = 0.0255
SAMPLE_SIZE = 10
NAME = "999_603777.csv"
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
        MAX = d.max()
        MIN = d.min()
        # newDataFrame[c] = ((d - MIN) / (MAX - MIN)).tolist()
        newDataFrame[c] = d.tolist()
    # newDataFrame.to_csv('D:/dataset/raw_norm_data/'+NAME, index=False)
    train_data = np.array(newDataFrame, dtype=np.float32)  # np.ndarray()
    train_x_list = torch.from_numpy(train_data)  # list
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
    return data[10:len(data)-10]


file_path = "D:/dataset/raw_data/"+NAME
rawdata, traindata = read_csv_file_data(file_path)
max_price = rawdata.iloc[:, 1].max()
min_price = rawdata.iloc[:, 1].min()

# print(traindata)
traindata.normal_(0, 1)
traindata = traindata.sigmoid()
# print(traindata)
# for i in range(traindata.shape[1]):
#     print("mean:= ", traindata[:, i].mean())
#     print("var:= ", traindata[:, i].var())
DATA_DIM = traindata.shape[1]
HIDE_DIM = 3
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
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 8),
            nn.Tanh(),
            nn.Linear(8, _output_dim),   # 压缩成3个特征, 进行 3D 图像可视化
        )
        # 解压
        self.decoder = nn.Sequential(
            nn.Linear(_output_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 32),
            nn.Tanh(),
            nn.Linear(32, 128),
            nn.Tanh(),
            nn.Linear(128, _input_dim),
            nn.Sigmoid(),       # 激励函数让输出值在 (0, 1)
        )

    def forward(self, x):
        x = self.dropout(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def predict(self, x):
        # x = self.encoder(x).detach()
        return self.encoder(x).detach()
        # return self.encoder(x).detach()


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
# autoencoder = Auto_Encoder(_input_dim=DATA_DIM, _hide_dim=HIDE_DIM)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()


def learn():
    loss_temp = []
    epoch = 0
    autoencoder.train()
    count = 0
    while count < 5:  # at least count five time smaller than BIAS
        epoch += 1
        loss_sum = 0
        step = 0
        for _, (x) in enumerate(trainLoader):
            step += 1
            x = torch.tensor(x)
            b_x = x.view(-1, DATA_DIM)
            b_y = x.view(-1, DATA_DIM)
            encoded, decoded = autoencoder(b_x)
            loss = loss_func(decoded, b_y)      # mean square error
            loss_sum += loss.data.item()
            # if loss.data.item() < BIAS:
            #     loss_temp.append(loss)
            # else:
            #     loss_temp.clear()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if loss_sum/step <= BIAS:
            count += 1
        else:
            count = 0
        if epoch % 100 == 0:
            print('epoch: '+str(epoch)+' loss : '+str(loss.data.item()))
    print('epoch:= '+str(epoch))


def predict():
    autoencoder.eval()
    result = []
    row = []
    res = []
    for item in range(len(traindata)):
        temp = traindata[item].unsqueeze(0)
        _, tt2 = autoencoder(temp)
        tt = torch.softmax(autoencoder.predict(temp), dim=1)
        tt = torch.squeeze(tt)
        tt2 = tt.detach()
        tt2 = torch.squeeze(tt2)
        res.append(tt.numpy())
        result.append(tt2.numpy()[1]*(max_price-min_price)+min_price)
        row.append(traindata[item][1]*(max_price-min_price)+min_price)

    file_path2 = "D:/dataset/processed_data/"+str(max_price)+"_"+str(min_price)+"_new3_"+NAME
    result_1 = pd.DataFrame(data=res)
    result_1.to_csv(file_path2, index=False)
    draw_data(row)
    draw_data(result, "out_put data")


learn()
predict()

