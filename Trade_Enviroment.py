import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import math
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
想办法怎么处理好前三维度和后面三个维度的影响力关系，不能只依靠股票信息来决定输出，毕竟一方面大多数情况
下股票信息波动较小，另一方面不能指望在同一下股票信息下个人持股量不同就可以训练产生不同的动作信息
"""
TRAIN_FILE_NAME = "37.551_10.55_1038_600977"
PRICE_FILE_NAME = "1038_600977"
# NAME = "1038_600977"
# dp = pd.read_csv("D:/dataset/raw_data/"+NAME+'.csv')
# dp = pd.read_csv("D:/dataset/processed_data/"+TRAIN_FILE_NAME+'.csv')
dp = pd.read_csv("D:/dataset/raw_norm_data/"+TRAIN_FILE_NAME+'.csv')
# dp = pd.read_csv("D:/dataset/raw_data/"+PRICE_FILE_NAME+'.csv')
dp = dp.iloc[20:, 1:]
dp2 = pd.read_csv("D:/dataset/raw_data/"+PRICE_FILE_NAME+'.csv')
dp2 = dp2.iloc[20:, 2]
# max_price = float(dp2.max())
# min_price = float(dp2.min())
# max_price = dp.iloc[:, 1].max()
# min_price = dp.iloc[:, 1].min()

hold = 1000
balance = 10000
data_num = dp.shape[0]
env_dim = dp.shape[1]
train_length = round(len(dp)*0.75)
Gamma = 0.9
k = 10


class Environment(nn.Module):
    def __init__(self):
        super(Environment, self).__init__()
        self.action_dim = env_dim
        self.balance = balance
        self.hold = hold
        # self.data = np.array(dp)
        # self.data = torch.from_numpy(dp.values).to(device)
        self.data = torch.tensor(dp.values, dtype=torch.float32).to(device)
        self.price_info = torch.tensor(dp2.values, dtype=torch.float32)
        self.env_length = len(self.data)
        self.train_length = train_length
        self.env_dim = env_dim
        self.index = 0
        self.reward = 0
        # self.max_value = max_price
        # self.min_value = min_price
        self.name = PRICE_FILE_NAME
        self.change_balance = 0
        self.change_hold = 0

    def step(self, _action):
        _action = _action.data.item()
        if _action < 0:
            _action = -1*round(abs(_action)*100)
        else:
            _action = round(_action*100)
        action = max(-50, min(50, _action))
        # print(action)
        new_hold = self.hold+action
        # price = self.data[self.index % self.train_length][1]*(max_price-min_price)+min_price
        price = self.price_info[self.index % self.train_length]
        new_balance = self.balance-(new_hold-self.hold)*price
        self.index += 1
        new_price = self.price_info[self.index % self.train_length]
        last_value = self.balance + self.hold * price
        if new_hold > 0 and new_balance > 0:
            # new_price = self.data[self.index % self.train_length][1]*(max_price-min_price)+min_price
            # r = new_balance+new_hold*new_price-last_value
            self.change_hold = new_hold-self.hold
            self.change_balance = new_balance-self.balance
            self.balance = new_balance
            self.hold = new_hold
        r = self.balance + self.hold * new_price - last_value
        new_state = self.state_cat(r, last_value)
        return new_state, r

    def test_step(self, action):
        action = int(action)
        new_hold = self.hold+action
        # price = self.data[self.index][1]*(max_price-min_price)+min_price
        price = self.price_info[self.index]
        new_balance = self.balance-(new_hold-self.hold)*price
        r = 0
        self.index += 1
        last_value = self.balance+self.hold*price
        if new_hold > 0 and new_balance > 0:
            # new_price = self.data[self.index][1]*(max_price-min_price)+min_price
            new_price = self.price_info[self.index]
            r = new_balance+new_hold*new_price-last_value
            self.change_hold = new_hold - self.hold
            self.change_balance = new_balance - self.balance
            self.balance = new_balance
            self.hold = new_hold
        new_state = self.state_cat(r, last_value)
        # new_state = np.concatenate((np.array([balance]), np.array([hold]), self.data[self.index]), axis=-1)
        return new_state, r

    def reset(self):
        self.hold = hold
        self.balance = balance
        self.index = 0
        self.change_balance = 0
        self.change_hold = 0
        price = self.price_info[self.index]
        last_value = self.balance + self.hold * price
        return self.state_cat(0, last_value)

    def state_cat(self, r, last_value):
        a = self.data[self.index]
        # b = r/last_value
        # c = self.change_balance/self.balance
        # d = self.change_hold/self.hold
        # norm = math.exp(b) + math.exp(c) + math.exp(d)
        b = torch.tensor([r/last_value], dtype=torch.float32)
        c = torch.tensor([self.change_balance/self.balance], dtype=torch.float32)
        d = torch.tensor([self.change_hold/self.hold], dtype=torch.float32)
        f = torch.cat((b, c, d), dim=0)
        e = torch.softmax(f, dim=0)
        # norm = math.exp(b)+math.exp(c)+math.exp(d)
        # norm = torch.tensor(norm)
        new_state = torch.cat((a, e), dim=0)
        return new_state

    def reset_for_test(self, index):
        self.index = index
        self.hold = hold
        self.balance = balance
        self.change_balance = 0
        self.change_hold = 0
        price = self.price_info[self.index]
        last_value = self.balance + self.hold * price
        return self.state_cat(0, last_value)


# env = Environment()
# state = env.reset()
# print(state)
# for i in range(10):
#     __action = np.random.randint(-10, 10)
#     state, _r = env.step(__action)
#     print(state)
#     print(_r)
