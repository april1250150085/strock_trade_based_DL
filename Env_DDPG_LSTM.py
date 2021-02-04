import numpy as np
import pandas as pd
import sys
import random
import torch
from matplotlib import pyplot as plt
import torch.utils.data as _data


def own_loader(x):
    x_tensor = torch.FloatTensor(x)
    return x_tensor


class MyDataset2(_data.Dataset):
    def __init__(self, datavector, loader=own_loader):
        self.datavector = datavector
        self.loader = loader

    def __getitem__(self, item):
        datatemp = self.datavector[item]
        ids = self.loader(datatemp)
        return ids

    def __len__(self):
        # return (self.datavector.size()[0])
        return len(self.datavector)


sequence_length = 10
TRAIN_FILE_NAME = ""
PRICE_FILE_NAME = ""
dp = pd.read_csv("D:/dataset/processed_data/"+TRAIN_FILE_NAME+'.csv')
dp = dp.iloc[20:, :]
dp2 = pd.read_csv("D:/dataset/raw_data/"+PRICE_FILE_NAME+'.csv')
dp2 = dp2.iloc[20+sequence_length:, 2]
state_dim = dp.shape[1]

# epochs = 2000
# data_dim = 6
# Batch_size = 128


def draw(_price):
    plt.figure(1)
    plt.plot(_price)
    plt.title("stock price")
    plt.show()


def create_dataset(data_set, look_back=sequence_length):
    dataX = []
    for i in range(len(data_set) - look_back):
        a = data_set[i:(i + look_back)]
        dataX.append(a)
    return torch.tensor(dataX)


data_X = create_dataset(dp)
dataset_length = len(data_X)
train_length = round(len(data_X)*0.75)
dataset = MyDataset2(data_X)


hold = 100
balance = 10000


class Environment(object):
    def __init__(self):
        self.action_dim = 1
        self.sequence_length = sequence_length
        self.state_dim = state_dim
        self.balance = balance
        self.hold = hold
        self.data = torch.tensor(data_X, dtype=torch.float32)  # 数据切片
        self.price_info = torch.tensor(dp2.values, dtype=torch.float32)
        self.index = 0
        self.reward = 0
        self.env_length = len(self.data)
        self.train_length = train_length
        self.index = 0
        self.reward = 0
        # self.max_value = max_price
        # self.min_value = min_price
        self.name = PRICE_FILE_NAME
        self.change_balance = 0
        self.change_hold = 0

    def step(self, _action):
        action = max(-50, min(50, int((0.1 + _action) * 10)))
        new_hold = self.hold + action
        # price = self.data[self.index % self.train_length][1]*(max_price-min_price)+min_price
        price = self.price_info[self.index % self.train_length]
        new_balance = self.balance - (new_hold - self.hold) * price
        r = 0
        self.index += 1
        if new_hold > 0 and new_balance > 0:
            # new_price = self.data[self.index % self.train_length][1]*(max_price-min_price)+min_price
            new_price = self.price_info[self.index % self.train_length]
            r = new_balance + new_hold * new_price - self.balance - self.hold * price
            self.change_hold = new_hold - self.hold
            self.change_balance = new_balance - self.balance
            self.balance = new_balance
            self.hold = new_hold
        new_state = self.state_cat(r)
        return new_state, r

    def test_step(self, action):
        action = int(action)
        new_hold = self.hold + action
        # price = self.data[self.index][1]*(max_price-min_price)+min_price
        price = self.price_info[self.index]
        new_balance = self.balance - (new_hold - self.hold) * price
        r = 0
        self.index += 1
        if new_hold > 0 and new_balance > 0:
            # new_price = self.data[self.index][1]*(max_price-min_price)+min_price
            new_price = self.price_info[self.index]
            r = new_balance + new_hold * new_price - self.balance - self.hold * price
            self.change_hold = new_hold - self.hold
            self.change_balance = new_balance - self.balance
            self.balance = new_balance
            self.hold = new_hold
        new_state = self.state_cat(r)
        # new_state = np.concatenate((np.array([balance]), np.array([hold]), self.data[self.index]), axis=-1)
        return new_state, r

    def reset(self):
        self.hold = hold
        self.balance = balance
        self.index = 0
        self.change_balance = 0
        self.change_hold = 0
        return self.state_cat(0)

    def state_cat(self, r):
        a = self.data[self.index]
        b = torch.tensor([r / (self.hold * self.price_info[self.index] + self.balance)], dtype=torch.float32)
        c = torch.tensor([self.change_balance / self.balance], dtype=torch.float32)
        d = torch.tensor([self.change_hold / self.hold], dtype=torch.float32)
        new_state = torch.cat((a, b, c, d), dim=0)
        return new_state

    def reset_for_test(self, index):
        self.index = index
        self.hold = hold
        self.balance = balance
        self.change_balance = 0
        self.change_hold = 0
        return self.state_cat(0)


# enev = Environment()
# s = enev.reset()
# state, state1, r = enev.step(0.5)
# # # temp = 10 * (np.random.random(data_dim)-0.5)
# # r = enev.step(torch.Tensor([0.1]))
# print(r)
# enev.reset()
# print(enev.balance)
# a = np.random.random([10, 2])
# print(a)

# import torch
# for i in range(10):
#     a = torch.randint(-1, 2, (1, 1)).data.item()
#     print(a)
