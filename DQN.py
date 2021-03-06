import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from Env_DQN import Environment
from Net import Net
from matplotlib import pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
BATCH_SIZE = 128  # 每一批的训练量
LR = 0.01  # 学习率
TAU = 0.001
EPSILON = 0.9  # 贪婪策略指数，Q-learning的一个指数，用于指示是探索还是利用。
GAMMA = 1  # reward discount
MEMORY_CAPACITY = 10000
env = Environment()
Hidden_num = 100
N_ACTIONS = env.action_dim
N_STATES = env.state_dim+2


# 创建Q-learning的模型
class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS, Hidden_num), Net(N_STATES, N_ACTIONS, Hidden_num)

        self.learn_step_counter = 0  # 如果次数到了，更新target_net
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化记忆
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    # 选择动作
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:  # 贪婪策略
            actions_value = self.eval_net.forward(x, False).detach()
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    # 存储记忆
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))  # 将每个参数打包起来
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s)  # 按照动作以列为对象进行索引，这样就知道当时采取的那个动作的Q值了
        q_eval = torch.gather(q_eval, 1, b_a)
        q_target = self.target_net(b_s_).detach()  # detach的作用就是不反向传播去更新，因为target的更新在前面定义好了的
        y = b_r + GAMMA * q_target.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data.item()


def draw(_loss, _reward):

    # plt.suptitle(u"TD-loss与累积收益随迭代次数变化趋势")
    plt1 = plt.subplot(2, 1, 1)
    plt1.set_title(u"TD-loss随迭代次数变化趋势")
    plt.xlabel(u"迭代次数")
    plt.ylabel(u"TD-loss")
    plt.plot(_loss)
    plt2 = plt.subplot(2, 1, 2)
    plt2.set_title(u"累积收益随迭代次数变化趋势")
    plt.xlabel(u"迭代次数")
    plt.ylabel(u"累积收益")
    plt.plot(_reward)
    plt.show()


def draw_loss(_loss):
    plt.title(u"损失函数随迭代次数变化趋势")
    plt.xlabel(u"迭代次数")
    plt.ylabel(u"TD-loss")
    plt.plot(_loss)
    plt.show()


def draw_reward(_reward):
    plt.title("累积收益随迭代次数变化趋势")
    plt.xlabel("迭代次数")
    plt.ylabel("累积收益")
    plt.plot(_reward)
    plt.show()

# dqn = DQN()


def train():
    dqn = DQN()
    loss_set = []
    reward = []
    for i_episode in range(400):
        s = env.reset()
        ep_r = 0
        loss_ = 0
        for j in range(env.train_length):
            a = dqn.choose_action(s)
            s_, r, r2 = env.step(a)
            r = r/(env.balance+env.hold*env.current_price+1)

            dqn.store_transition(s, a, r, s_)

            ep_r += r2
            if dqn.memory_counter > MEMORY_CAPACITY:
                __loss = dqn.learn()
                loss_set.append(__loss)
                loss_ += __loss
            s = s_
        # print(loss_)
        if dqn.memory_counter > MEMORY_CAPACITY:
            reward.append(ep_r)
        print("episode:= ", i_episode, "  reward:= ", ep_r.data.item(), "   loss:= ", loss_/env.train_length)
        if i_episode % 10 == 0 and dqn.memory_counter > MEMORY_CAPACITY:
            # draw(loss_set, reward)
            # draw_loss(loss_set)
            # draw_reward(reward)
            torch.save(dqn, 'model2/dqn/new_norm_dqn_'+str(i_episode)+'.pt')


# train()
