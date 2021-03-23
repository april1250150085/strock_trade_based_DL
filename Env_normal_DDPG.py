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
FILE_NAME = "1000_603067"

# dp = pd.read_csv("D:/dataset/norm_data/" + FILE_NAME + '.csv')
dp = pd.read_csv("D:/dataset/new_processed/"+FILE_NAME+'.csv')
dp = dp.iloc[20:, 1:]
_dp = pd.read_csv("D:/dataset/raw_data/" + FILE_NAME + '.csv')
dp2 = _dp.iloc[20:, 2]


max_price = dp2.values.max()
min_price = dp2.values.min()
hold = 100
balance = 10000
data_num = dp.shape[0]
state_dim = dp.shape[1]
action_dim = 1
train_length = round(len(dp) * 0.75)


class Environment(nn.Module):
    def __init__(self):
        super(Environment, self).__init__()
        self.action_dim = action_dim
        self.balance = balance
        self.hold = hold
        self.data = torch.tensor(dp.values, dtype=torch.float32).to(device)
        self.price_info = torch.tensor(dp2.values, dtype=torch.float32)
        self.env_length = len(self.data)
        self.train_length = train_length
        self.state_dim = state_dim
        self.index = 0
        self.reward = 0
        self.max_price = max_price
        self.min_price = min_price
        self.current_price = self.price_info[self.index]
        self.last_price = 0
        self.name = FILE_NAME
        self.change_balance = 0
        self.change_hold = 0

    def step(self, _action):  # 0->sell, 1->hold, 2->buy
        self.last_price = self.current_price
        action = round(_action)
        last_hold = self.hold
        last_balance = self.balance
        price = self.price_info[self.index % self.train_length]
        new_hold = self.hold + action
        new_balance = self.balance - (new_hold - self.hold) * price * 100
        self.index += 1
        new_price = self.price_info[self.index % self.train_length]
        self.current_price = new_price
        last_value = self.balance + self.hold * price * 100
        if new_hold > 0 and new_balance > 0:
            self.change_hold = new_hold - self.hold
            self.change_balance = new_balance - self.balance
            self.balance = new_balance
            self.hold = new_hold
        r = self.balance + self.hold * new_price * 100 - last_value
        r2 = r
        a = torch.tensor([(self.hold - last_hold) / (1 + last_hold)])
        b = torch.tensor([(self.balance - last_balance) / (1 + last_balance)])

        new_state = torch.cat((a, b, self.data[self.index]), dim=0)
        return new_state, r, r2

    def test_step(self, _action):
        last_hold = self.hold
        last_balance = self.balance
        price = self.price_info[self.index]
        new_hold = self.hold + _action
        new_balance = self.balance - (new_hold - self.hold) * price * 100
        self.index += 1
        new_price = self.price_info[self.index]
        self.current_price = new_price
        # print(price.data.item(), " -> ", new_price.data.item())
        last_value = self.balance + self.hold * price * 100
        # r2 = (new_price - price) * self.hold * 100
        if new_hold > 0 and new_balance > 0:
            self.change_hold = new_hold - self.hold
            self.change_balance = new_balance - self.balance
            self.balance = new_balance
            self.hold = new_hold
        r = self.balance + self.hold * new_price * 100 - last_value
        a = torch.tensor([(self.hold - last_hold) / (1 + last_hold)])
        b = torch.tensor([(self.balance - last_balance) / (1 + last_balance)])
        r2 = r
        new_state = torch.cat((a, b, self.data[self.index]), dim=0)
        return new_state, r, r2

    def reset(self):
        self.hold = hold
        self.balance = balance
        self.index = 0
        self.change_balance = 0
        self.change_hold = 0
        self.current_price = self.price_info[self.index]
        a = torch.tensor([0 / (1 + self.hold)])
        b = torch.tensor([0 / (1 + self.balance)])

        new_state = torch.cat((a, b, self.data[self.index]), dim=0)
        return new_state

    def state_cat(self, r, last_value):
        a = self.data[self.index]
        b = torch.tensor([r / last_value], dtype=torch.float32)
        c = torch.tensor([self.change_balance / self.balance], dtype=torch.float32)
        d = torch.tensor([self.change_hold / self.hold], dtype=torch.float32)
        f = torch.cat((b, c, d), dim=0)
        e = torch.softmax(f, dim=0)
        new_state = torch.cat((a, e), dim=0)
        return new_state

    def reset_for_test(self, index):
        self.hold = hold
        self.balance = balance
        self.index = index
        self.change_balance = 0
        self.change_hold = 0
        a = torch.tensor([0 / (1 + self.hold)])
        b = torch.tensor([0 / (1 + self.balance)])
        self.current_price = self.price_info[self.index]
        new_state = torch.cat((a, b, self.data[self.index]), dim=0)
        return new_state

# env = Environment()
# state = env.reset()
# print(state)
# for i in range(20):
#     __action = np.random.randint(0, 3)
#     print("--------------------")
#     print("old hold:= ", env.hold, " balance:= ", env.balance)
#     print("action:= ", __action-1)
#     state, _r = env.step(__action)
#     print("new hold:= ", env.hold, " balance:= ", env.balance.data.item())
#     print("r:= ", _r.data.item())
#     # print(state)
#     # print(_r.data.item())
