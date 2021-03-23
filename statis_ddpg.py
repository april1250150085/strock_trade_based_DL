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
# from lstm_reg_simple import Net as Net


from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
env = Environment()
flag = False
price_recode = list()
x_index = list()
# name = 'Advance DQN'
name = 'norm-filter DDPG'

# MODEL_NAME = 'model2/ddpg/AE_ddpg_50.pt'
# MODEL_NAME = 'model2/ddpg/normal_ddpg_390.pt'
# MODEL_NAME = 'model2/ddpg/norm_LSTM_ddpg_20.pt'
# MODEL_NAME = 'model2/ddpg/AE_LSTM_ddpg_20.pt'
MODEL_NAME = 'model2/ddpg/AE_filter_ddpg_30.pt'
# MODEL_NAME = 'model2/ddpg/norm_filter_ddpg_490.pt'
model = torch.load(MODEL_NAME)
total_reward = 0
start_index = env.train_length
# start_index = 0
test_time = 10
model_reward_recode = np.zeros((env.env_length - 1 - start_index))
model_reward = np.zeros(test_time)
random_reward_recode = np.zeros((env.env_length - 1 - start_index))
random_reward = np.zeros(test_time)
one_reward_recode = np.zeros((env.env_length - 1 - start_index))
one_reward = np.zeros(test_time)

for i in range(test_time):
    s = env.reset_for_test(start_index)
    rew = 0
    for index in range(start_index, env.env_length - 1):
        action = model.choose_action(s)
        # print('-----------', action, '-----------')
        # action = np.clip(round(action.data.item()+np.random.uniform(-8, 8)), -10, 10)
        # action = np.random.randint(0, 3)
        # action = 1
        # print(action)
        s_, reward, reward2 = env.test_step(action)
        total_reward += reward
        rew += reward
        model_reward_recode[index - start_index] += rew.data.item()
        # model_reward_recode[index - start_index] = max(model_reward_recode[index - start_index], rew.data.item())
        if not flag:
            x_index.append(index)
        s = s_
        # print("current reward:= ", rew)
    # print("reward: ", rew.data.item())
    model_reward[i] = rew.data.item()
    flag = True

for i in range(test_time):
    s = env.reset_for_test(start_index)
    rew = 0
    for index in range(start_index, env.env_length - 1):
        action = np.random.randint(-10, 11)
        s_, reward, reward2 = env.test_step(action)
        total_reward += reward
        rew += reward
        random_reward_recode[index - start_index] += rew.data.item()
        s = s_
    random_reward[i] = rew.data.item()

for i in range(test_time):
    s = env.reset_for_test(start_index)
    rew = 0
    for index in range(start_index, env.env_length - 1):
        action = 0
        s_, reward, reward2 = env.test_step(action)
        total_reward += reward
        rew += reward
        one_reward_recode[index - start_index] += rew.data.item()
        s = s_
    one_reward[i] = rew.data.item()

for index in range(start_index, env.env_length - 1):
    model_reward_recode[index - start_index] /= test_time
    random_reward_recode[index - start_index] /= test_time
    one_reward_recode[index - start_index] /= test_time

env.reset_for_test(start_index)
origin = env.balance + env.hold * env.current_price * 100.

model_ratio = model_reward_recode[-1] / origin * 100.
random_ratio = random_reward_recode[-1] / origin * 100.
one_ratio = one_reward_recode[-1] / origin * 100.
print("标准差： ", round(model_reward.std()), "  , 均值", round(model_reward.mean()), " ,最大值",
      round(model_reward.max()), ", 最小值 ", round(model_reward.min()))
print("标准差： ", round(random_reward.std()), "  , 均值", round(random_reward.mean()), " ,最大值",
      round(random_reward.max()), ", 最小值 ", round(random_reward.min()))
print("标准差： ", round(one_reward.std()), "  , 均值", round(one_reward.mean()), " ,最大值",
      round(one_reward.max()), ", 最小值 ", round(one_reward.min()))
print("model ratio:= ", model_ratio.data.item())
print("random ratio:= ", random_ratio.data.item())
print("one action ratio:= ", one_ratio.data.item())
plt.title("不同动作控制方式下累积收益变化趋势")
plt.plot(x_index, model_reward_recode, label=name)
plt.plot(x_index, random_reward_recode, label="choose action randomly")
plt.plot(x_index, one_reward_recode, label="just hold")
plt.legend()
if model_ratio > 5:
    plt.show()
