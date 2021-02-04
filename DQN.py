import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from Env_DQN import Environment


BATCH_SIZE = 32  # 每一批的训练量
LR = 0.01  # 学习率
EPSILON = 0.9  # 贪婪策略指数，Q-learning的一个指数，用于指示是探索还是利用。
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target的更新频率
MEMORY_CAPACITY = 20
env = Environment()
# env = gym.make('CartPole-v0')
# env = env.unwrapped
N_ACTIONS = env.action_dim
N_STATES = env.env_dim+3
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(),
#                               int) else env.action_space.sample().shape  # to confirm the shape


# 创建神经网络模型，输出的是可能的动作
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


# 创建Q-learning的模型
class DQN(object):
    def __init__(self):
        # 两张网是一样的，不过就是target_net是每100次更新一次，eval_net每次都更新
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # 如果次数到了，更新target_net
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化记忆
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    # 选择动作
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:  # 贪婪策略
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
            # if ENV_A_SHAPE == 0:
            #     action = action[0]
            # else:
            #     action.reshape(ENV_A_SHAPE) # return the argmax index
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
            # if ENV_A_SHAPE == 0:
            #     action = action
            # else:
            #     action.reshape(ENV_A_SHAPE)
            # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    # 存储记忆
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))  # 将每个参数打包起来
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 学习过程
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach的作用就是不反向传播去更新，因为target的更新在前面定义好了的
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()


for i_episode in range(400):
    for j in range(100):
        s = env.reset()  # 搜集当前环境状态。
        ep_r = 0
        a = dqn.choose_action(s)
        s_, r = env.step(a)

        # modify the reward
        # x, x_dot, theta, theta_dot = s_
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # r = r1 + r2

        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            s = s_
