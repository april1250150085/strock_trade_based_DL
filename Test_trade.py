from Trade_Enviroment import Environment
import torch
from DDPG_stock_trade_backup import ANet
from DDPG_stock_trade_backup import CNet
from DDPG_stock_trade_backup import DDPG
import numpy as np


rew = 0
for i in range(0, 10):
    env = Environment()
    path = "model/ddpg_"+env.name+"_raw"+str(0)+".pt"
    # path = "model/ddpg_37.551_10.55_1038_600977_raw0.pt"
    s = env.reset_for_test(0)
    model = torch.load(path)
    reward = 0
    for i in range(0, env.env_length-1):
        # a = np.random.randint(-10, 10)
        a = model.choose_action(s)
        a = torch.clamp(torch.normal(a, 1), -1, 1)
        # print(a)
        # a = 0
        s, r = env.test_step(a)
        reward += r
    print(reward)
    rew += reward
print(rew/10)
