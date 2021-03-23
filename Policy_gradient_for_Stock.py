import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from matplotlib import pyplot as plt
from Env_Policy_Gradient import Environment
import time

GAMMA = 1

env = Environment()
state_dim = env.state_dim + 2
action_dim = 3
hidden_dim = 128
print('observation space:', state_dim)
print('action space:', action_dim)


class PolicyGradient(nn.Module):
    def __init__(self):
        super(PolicyGradient, self).__init__()
        self.fc = nn.Linear(state_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.out = nn.Linear(hidden_dim, action_dim)

        self.prob_list = list()
        self.reward_list = list()

    def forward(self, x):
        x = self.fc(x)
        x = torch.relu(self.dropout(x))
        action = self.out(x)
        return torch.softmax(action, dim=1)



pg = PolicyGradient()
optimizer = optim.Adam(pg.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()  # 非负的最小值，使得归一化时分母不为0


def draw(loss_info, reward_info):
    plt.suptitle("loss_info and reward_info")
    plt1 = plt.subplot(2, 1, 1)
    plt1.set_title("loss information")
    plt.plot(loss_info)
    plt2 = plt.subplot(2, 1, 2)
    plt2.set_title("reward information")
    plt.plot(reward_info)
    plt.show()


def select_action(state):
    # 选择动作，这个动作不是根据Q值来选择，而是使用softmax生成的概率来选
    # 不需要epsilon-greedy，因为概率本身就具有随机性
    state = state.unsqueeze(0)
    raw_action = pg(state)
    m = Categorical(raw_action)  # 生成分布
    action = m.sample()  # 从分布中采样
    pg.prob_list.append(m.log_prob(action))  # 取对数似然 logπ(s,a)
    return action.item()  # 返回一个元素值


def finish_episode():
    accumulate_r = 0
    policy_loss = []
    returns = []
    for r in pg.reward_list[:]:
        # print(r)
        accumulate_r = r + GAMMA * accumulate_r
        returns.insert(0, accumulate_r)  # 将R插入到指定的位置0处, 蒙特卡洛方法
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)  # 归一化
    for log_prob, R in zip(pg.prob_list, returns):
        policy_loss.append(-log_prob * R)  # 梯度上升
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()  # 求和
    policy_loss.backward()
    optimizer.step()
    del pg.reward_list[::]  # 清空episode 数据
    del pg.prob_list[::]
    return policy_loss.data.item()


def main():
    loss = list()
    acc_reward = list()
    last_reward = 0
    ep_reward = 0
    count = 0
    for i_episode in range(1000):  # 采集（训练）最多1000个序列
        last_reward = ep_reward
        state, ep_reward = env.reset(), 0  # ep_reward表示每个episode中的reward
        t = 1
        for t in range(1, env.train_length):
            action = select_action(state)
            s_, r, r2 = env.step(action)
            r = r / (env.balance + env.hold * env.current_price + 1)
            pg.reward_list.insert(0, r)
            ep_reward += r
            state = s_
        acc_reward.append(ep_reward)
        loss.append(finish_episode())
        if i_episode % 10 == 0:
            print('Episode {}\tLast reward: {:.2f}\t'.format(
                i_episode, ep_reward))
        # if i_episode > 0 and i_episode % 50 == 0:
            # draw(loss, acc_reward)
        if last_reward == ep_reward:
            count += 1
            if count == 10:
                print("train time := ", i_episode)
                break
        if i_episode % 40 == 0:
            torch.save(pg, "model5/pg/" + str(env.name) + "_AE_pg" + str(i_episode) + ".pt")


if __name__ == '__main__':
    main()
