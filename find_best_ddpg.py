from Env_normal_DDPG import Environment
import torch
# from DDPG_for_Stock import DDPG
# from DDPG_for_Stock import ANet
# from DDPG_for_Stock import CNet
# from LSTM_DDPG_for_Stock import DDPG
# from LSTM_DDPG_for_Stock import ANet
# from LSTM_DDPG_for_Stock import CNet
from Filter_DDPG_for_Stock import DDPG
from Filter_DDPG_for_Stock import ANet
from Filter_DDPG_for_Stock import CNet
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
env = Environment()
flag = False
current = 20
price_recode = list()
x_index = list()
model_reward = 0
for ij in range(0, 48):
    current += 10
    print(current)
    # MODEL_NAME = 'model2/ddpg/normal_ddpg_'+str(current)+'.pt'
    # MODEL_NAME = 'model2/ddpg/AE_ddpg_'+str(current)+'.pt'
    # MODEL_NAME = 'model2/ddpg/norm_LSTM_ddpg_'+str(current)+'.pt'
    # MODEL_NAME = 'model2/ddpg/AE_LSTM_ddpg_'+str(current)+'.pt'
    # MODEL_NAME = 'model2/ddpg/AE_filter_ddpg_'+str(current)+'.pt'
    # MODEL_NAME = 'model2/ddpg/norm_filter_ddpg_'+str(current)+'.pt'
    model = torch.load(MODEL_NAME)
    total_reward = 0
    start_index = env.train_length
    # start_index = 0
    model_reward_recode = list()
    random_reward_recode = list()
    one_reward_recode = list()

    for i in range(1):
        s = env.reset_for_test(start_index)
        rew = 0
        for index in range(start_index, env.env_length - 1):
            action = model.choose_action(s)
            # print('-----------', action, '-----------')
            # action = np.clip(round(action+np.random.normal(0, 1)), 0, 2)
            # action = np.random.randint(0, 3)
            # action = 1
            # print(action)
            # if index == 700:
            #     print("a")
            s_, reward, reward2 = env.test_step(action)
            total_reward += reward.data.item()
            rew += reward
            model_reward_recode.append(rew.data.item())
            if not flag:
                price_recode.append(env.current_price)
                x_index.append(index)
            s = s_
            # print("current reward:= ", rew)
        # print("reward: ", rew.data.item())
        model_reward = rew
    flag = True
    for i in range(1):
        s = env.reset_for_test(start_index)
        rew = 0
        for index in range(start_index, env.env_length - 1):
            action = np.random.randint(0, 3)
            s_, _, reward = env.test_step(action)
            total_reward += reward
            rew += reward
            random_reward_recode.append(rew.data.item())
            s = s_
        # print("reward: ", rew)

    for i in range(1):
        s = env.reset_for_test(start_index)
        rew = 0
        for index in range(start_index, env.env_length - 1):
            action = 1
            s_, _, reward = env.test_step(action)
            total_reward += reward
            rew += reward
            one_reward_recode.append(rew.data.item())
            s = s_
        # print("reward: ", rew)
    # print(total_reward/1)
    # plt.show()
    reward_avg = total_reward / 10.
    env.reset_for_test(start_index)
    origin = env.balance + env.hold * env.current_price * 100.
    reward_ratio = model_reward / origin * 100.
    print("when train ", str(current), " ratio:= ", reward_ratio)
    if reward_ratio > 50:
        plt.title(str(current))
        plt.plot(x_index, model_reward_recode, label="model")
        # plt.show()
        plt.plot(x_index, random_reward_recode, label="random")
        # plt.show()
        plt.plot(x_index, one_reward_recode, label="hold")
        plt.legend()
        plt.show()
# plt.figure(2)
# plt.plot(x_index, price_recode)
# plt.show()
