import numpy as np
import pandas as pd
import torch
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NAME = "37.551_10.55_1038_600977"
dp = pd.read_csv("D:/dataset/processed_data/"+NAME+".csv")  # 608_300502.csv
max_price = float(NAME.split("_")[0])
min_price = float(NAME.split("_")[1])

hold = 20
balance = 10000
data_num = dp.shape[0]
env_dim = dp.shape[1]-1
train_length = round(len(dp)*0.75)
Gamma = 0.9
k = 10


class Environment(object):
    def __init__(self):
        self.action_dim = env_dim
        self.balance = balance
        self.hold = hold
        self.data = torch.tensor(dp.iloc[:, 1:].values, dtype=torch.float32).to(device)
        self.env_length = len(self.data)
        self.train_length = train_length
        self.env_dim = env_dim
        self.index = 0
        self.reward = 0
        self.max_value = max_price
        self.min_value = min_price
        self.name = NAME

    def step(self, action):
        action = int(action)
        new_hold = self.hold+action
        price = self.data[self.index % self.train_length][2]*(max_price-min_price)+min_price
        new_balance = self.balance-(new_hold-self.hold)*price
        r = 0
        self.index += 1
        if new_hold >= 0 and new_balance >= 0:
            new_price = self.data[self.index % self.train_length][2]*(max_price-min_price)+min_price
            r = new_balance+new_hold*new_price-self.balance-self.hold*price
            self.balance = new_balance
            self.hold = new_hold
        new_state = self.data[self.index % train_length]
        # new_state = np.concatenate((np.array([balance]), np.array([hold]), self.data[self.index]), axis=-1)
        return new_state, r

    def test_step(self, action):
        action = int(action)
        new_hold = self.hold+action
        price = self.data[self.index][2]*(max_price-min_price)+min_price
        new_balance = self.balance-(new_hold-self.hold)*price
        r = 0
        self.index += 1
        if new_hold >= 0 and new_balance >= 0:
            new_price = self.data[self.index][2]*(max_price-min_price)+min_price
            r = new_balance+new_hold*new_price-self.balance-self.hold*price
            self.balance = new_balance
            self.hold = new_hold
        new_state = self.data[self.index]
        # new_state = np.concatenate((np.array([balance]), np.array([hold]), self.data[self.index]), axis=-1)
        return new_state, r

    def reset(self):
        self.hold = hold
        self.balance = balance
        self.index = 0
        state = self.data[0]
        return state

    def reset_for_test(self, index):
        self.index = index
        self.hold = hold
        self.balance = balance
        state = self.data[index]
        return state



# env = Environment()
# state = env.reset()
# print(state)
# for i in range(100):
#     action = np.random.randint(-10, 10)
#     state, r = env.step(action)
#     # print(state)
#     print(r)
