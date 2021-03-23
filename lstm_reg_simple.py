import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import pandas as pd
from ast import literal_eval
import csv
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
Filename = "1001_601128"
input_data_dim = 1
sequence_length = 10
hidden_lstm_num = 32
hidden_fc_num = 128

# dp = pd.read_csv("D:/dataset/cont_data/" + Filename + "2.csv")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=input_data_dim, hidden_size=hidden_lstm_num)
        self.fc = nn.Linear(hidden_lstm_num * sequence_length, hidden_fc_num)
        self.reg = nn.Linear(hidden_fc_num, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.view(-1, hidden_lstm_num * sequence_length)
        x = torch.sigmoid(self.fc(x))
        out = torch.tanh(self.reg(x))
        return out


# max_v = dp.iloc[:, -1].max()
# min_v = dp.iloc[:, -1].min()
# data_x = list()
# for i in range(dp.shape[0]):
#     data_x.append((max_v - dp.values[i, -1]) / (max_v - min_v))
#
# data_set_x = []
# data_set_y = []
# len_tmp = len(data_x)-10
# data_temp = np.zeros((len_tmp, 10))
# for i in range(len(data_x) - 10):
#     temp = []
#     for j in range(10):
#         temp.append([data_x[i + j]])
#         data_temp[i][j] = data_x[i+j]
#     data_set_x.append(temp)
#     data_set_y.append(data_x[i + 10])
# #
# #
# reg_data = pd.DataFrame(data_temp, dtype=np.float)
# reg_data.to_csv("D:/dataset/regdata/" + Filename + ".csv", index=False)
#
#
def train():
    lstm = Net()
    optimizer = torch.optim.Adam(lstm.parameters())
    for i_episode in range(500):
        for j in range(len(data_set_x)):
            x = torch.tensor(data_set_x[j]).unsqueeze(0)
            x = lstm(x)
            y = torch.tensor(data_set_y[j])
            loss = nn.MSELoss()(x, y)
            lstm.zero_grad()
            loss.backward()
            optimizer.step()
        if i_episode % 50 == 0 and i_episode > 0:
            print(i_episode, "  loss:= ", loss.data.item())
            result = list()
            real = list()
            row = list()
            for i in range(len(data_set_x)):
                x = torch.tensor(data_set_x[i]).unsqueeze(0)
                result.append(max_v - (max_v - min_v) * lstm(x).data.item())
                real.append(max_v - (max_v - min_v) * data_set_y[i])
                row.append(i)

            plt.figure(1)
            plt.title(u'训练次数' + str(i_episode))
            plt.plot(row, result, label="raw data")
            # plt.figure(2)
            plt.plot(row, real, '-r', label="output data")
            plt.legend()
            plt.show()
            torch.save(lstm, 'D:/hjl_python_code/My_own_ddpg/model4/lstm_reg/'+Filename+'_lstm_reg_simple.pt')
    # print("a")


# train()


def test():
    model = torch.load('D:/hjl_python_code/My_own_ddpg/model4/lstm_reg/lstm_reg_simple2.pt')
    max_error = 0
    correct = 0
    regdata = pd.read_csv("D:/dataset/regdata/" + Filename + ".csv")
    for i in range(0, regdata.shape[0]):
        # x = torch.tensor(regdata.values[i]).unsqueeze(0)
        a = data_set_x[i]
        b = regdata.values[i, :]
        x_ = torch.tensor(b).unsqueeze(0)
        x_ = x_.unsqueeze(2)
        x_ = torch.tensor(x_, dtype=torch.float32)
        output = model(x_).data.item()
        # output = max_v - (max_v - min_v) * model(x_).data.item()
        real = data_set_y[i]
        # real = max_v - (max_v - min_v) * data_set_y[i]
        max_error = max(max_error, abs(real - output))
        if abs(real - output) <= 2:
            correct += 1
        # max_error = max(max_error, abs(real-output)/real)
        # last_price = max_v - (max_v - min_v) * data_set_y[i - 1]
        # if (output - last_price) * (real - last_price) >= 0:
        #     correct += 1
    print(correct)
    print(correct / (len(data_set_x)))
    # print(max_error / len(data_set_x))
    print(max_error)


# test()
