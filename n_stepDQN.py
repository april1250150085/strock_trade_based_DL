import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from Env_DQN import Environment
from matplotlib import pyplot as plt
from Net import Net
BATCH_SIZE = 64  # 每一批的训练量
LR = 0.01  # 学习率
TAU = 0.001
EPSILON = 0.9  # 贪婪策略指数，Q-learning的一个指数，用于指示是探索还是利用。
GAMMA = 1  # reward discount
MEMORY_CAPACITY = 10000
env = Environment()
Hidden_num = 128
N_ACTIONS = env.action_dim
N_STATES = env.state_dim+2
T = MEMORY_CAPACITY-1
N = 5


# 创建Q-learning的模型
class DQN(object):
    def __init__(self):
        # 两张网是一样的，不过就是target_net是每100次更新一次，eval_net每次都更新
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS, Hidden_num), Net(N_STATES, N_ACTIONS, Hidden_num)

        self.learn_step_counter = 0  # 如果次数到了，更新target_net
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化记忆
        self.memory_state = np.zeros((MEMORY_CAPACITY, N_STATES))
        self.memory_next_state = np.zeros((MEMORY_CAPACITY, N_STATES))
        self.memory_action = np.zeros((MEMORY_CAPACITY, 1))
        self.memory_reward = np.zeros((MEMORY_CAPACITY, 1))
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
        self.memory_state[index, :] = s
        self.memory_next_state[index, :] = s_
        self.memory_action[index, :] = a
        self.memory_reward[index, :] = r
        # self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
        self.learn_step_counter += 1

        # 学习过程
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        bound = min(T - sample_index.max(), N)-1
        s_memory = list()
        next_s_memory = list()
        a_memory = list()
        r_memory = list()
        for index in sample_index:
            r_memory.append(self.memory_reward[index:index+bound+1, ])
        s_memory.append(self.memory_state[sample_index])
        next_s_memory.append(self.memory_next_state[sample_index+bound])
        a_memory.append(self.memory_action[sample_index])

        # b_memory = self.memory[sample_index, :]
        b_s = torch.tensor(s_memory, dtype=torch.float32).squeeze()
        b_s_ = torch.tensor(next_s_memory, dtype=torch.float32).squeeze()
        b_r = torch.tensor(r_memory, dtype=torch.float32)
        b_r = torch.sum(b_r, dim=1)
        b_a = torch.tensor(a_memory, dtype=torch.long).squeeze(0)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s)
        q_eval = q_eval.gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach的作用就是不反向传播去更新，因为target的更新在前面定义好了的
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data.item()


    def learn2(self):
        sample_index = np.random.choice(self.memory_counter, BATCH_SIZE)
        s_memory = list()
        next_s_memory = list()
        a_memory = list()
        r_memory = list()
        r_memory.append(self.memory_reward[sample_index])
        s_memory.append(self.memory_state[sample_index])
        next_s_memory.append(self.memory_next_state[sample_index])
        a_memory.append(self.memory_action[sample_index])


        b_s = torch.tensor(s_memory, dtype=torch.float32).squeeze()
        b_s_ = torch.tensor(next_s_memory, dtype=torch.float32).squeeze()
        b_r = torch.tensor(r_memory, dtype=torch.float32)
        b_r = torch.sum(b_r, dim=1)
        b_a = torch.tensor(a_memory, dtype=torch.long).squeeze(0)


        q_eval = self.eval_net(b_s)
        q_eval = q_eval.gather(1, b_a)  # shape (batch, 1)
        action = torch.argmax(self.eval_net(b_s_), dim=1).view(BATCH_SIZE, 1)
        q_next = self.target_net(b_s_).gather(dim=1, index=action).detach()  # detach的作用就是不反向传播去更新，因为target的更新在前面定义好了的
        q_target = b_r + GAMMA * q_next.view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data.item()


def draw(_loss, _reward):
    plt.suptitle("loss_info and reward_info")
    plt1 = plt.subplot(2, 1, 1)
    plt1.set_title("loss information")
    plt.plot(_loss)
    plt2 = plt.subplot(2, 1, 2)
    plt2.set_title("reward information")
    plt.plot(_reward)
    plt.show()


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
            elif dqn.memory_counter > 5*BATCH_SIZE:
                dqn.learn2()
            s = s_
        # print(loss_)
        if dqn.memory_counter > MEMORY_CAPACITY:
            reward.append(ep_r)
        print("episode:= ", i_episode, "  reward:= ", ep_r.data.item(), "   loss:= ", loss_/env.train_length)
        if i_episode % 10 == 0 and dqn.memory_counter > MEMORY_CAPACITY:
            # draw(loss_set, reward)
            torch.save(dqn, 'model2/dqn/advance_dqn/AE_n_stepDQN_'+str(i_episode)+'.pt')


# train()
