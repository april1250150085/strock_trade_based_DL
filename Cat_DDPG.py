'''
torch = 0.41
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
from matplotlib import pyplot as plt
from Trade_Enviroment import Environment
import warnings

#####################  hyper parameters  ####################

MAX_EPISODES = 500
MAX_EP_STEPS = 200
LR_A = 0.01  # learning rate for actor
LR_C = 0.02  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
RENDER = False
ENV_NAME = 'Pendulum-v0'


###############################  DDPG  ####################################


class ANet(nn.Module):  # ae(s)=a
    def __init__(self, _s_dim, _a_dim):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(_s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(30, _a_dim)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        x = torch.tanh(x)
        actions_value = x * 2
        return actions_value


class CNet(nn.Module):  # ae(s)=a
    def __init__(self, _s_dim, _a_dim):
        super(CNet, self).__init__()
        self.fcs = nn.Linear(_s_dim, 30)
        self.fcs.weight.data.normal_(0, 0.1)  # initialization
        self.fca = nn.Linear(_a_dim, 30)
        self.fca.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, _s, _a):
        x = self.fcs(_s)
        y = self.fca(_a)
        net = F.relu(x + y)
        actions_value = self.out(net)
        return actions_value


class Net(nn.Module):
    def __init__(self, state_dim):
        super(Net, self).__init__()
        self.state_dim = state_dim
        # self.lstm_s = nn.LSTM(s_dim, 10, 2)
        self.lstm_s = nn.Linear(s_dim, 10)
        self.get_action1 = nn.Linear(10, 30)
        self.get_action2 = nn.Linear(30, 1)
        self.get_action_features = nn.Linear(1, 30)
        self.get_value = nn.Linear(10, 30)
        self.get_Q_Value = nn.Linear(30, 1)

    def forward(self, s, action=0, get_action=False, set_action=False):
        # x, _ = self.lstm_s(s)
        # x = x.view(-1, 10)
        x = self.lstm_s(s)
        s_tmp = torch.relu(x)
        if set_action:
            v_value = self.get_value(s_tmp)
            action_value = self.get_action_features(action)
            q_value = self.get_Q_Value(torch.relu(action_value + v_value))
            return q_value
        _action = torch.relu(self.get_action1(s_tmp))
        _action = torch.tanh(self.get_action2(_action)) * 2
        if get_action:
            return _action
        v_value = self.get_value(s_tmp)
        action_value = self.get_action_features(_action)
        tt = v_value + action_value
        q_value = self.get_Q_Value(torch.relu(v_value + action_value))
        return q_value


class DDPG(object):
    def __init__(self, a_dim=0, s_dim=0, a_bound=0, ):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.evalNet = Net(s_dim)
        self.targetNet = Net(s_dim)
        self.evaltrain = torch.optim.Adam(self.evalNet.parameters(), lr=LR_C)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        # s = s.unsqueeze(0)
        return self.evalNet(s, get_action=True, set_action=False)[0].detach()

    def learn(self):
        for target_param, param in zip(self.targetNet.parameters(), self.evalNet.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])

        q = self.evalNet(bs)  # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
        loss_a = -torch.mean(q)
        self.evaltrain.zero_grad()
        loss_a.backward()
        self.evaltrain.step()
        q_ = self.targetNet(bs_).detach()  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = br + GAMMA * q_  # q_target = 负的
        q_v = self.evalNet(bs, ba, False, True)
        td_error = self.loss_td(q_target, q_v)
        self.evalNet.zero_grad()
        td_error.backward()
        self.evaltrain.step()
        return -loss_a.data.item(), td_error.data.item()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1


env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]


# a_bound = env.action_space.high


def draw(_q, _td):
    q = np.array(_q)
    td = np.array(_td)
    plt.figure(1)
    plt.title(u"Cat-DDPG算法中Q_value随迭代次数变化曲线", fontproperties='SimHei')
    plt.xlabel(u"迭代次数", fontproperties='SimHei')
    plt.ylabel(u"Q-value信息", fontproperties='SimHei')
    plt.plot(q)
    plt.show()
    plt.figure(2)
    plt.title(u"Cat-DDPG算法中TD-error随迭代次数变化曲线", fontproperties='SimHei')
    plt.xlabel(u"迭代次数", fontproperties='SimHei')
    plt.ylabel(u"TD-error信息", fontproperties='SimHei')
    plt.plot(td)
    plt.show()


ddpg = DDPG(a_dim, s_dim)


def train():
    RENDER = False
    var = 3  # control exploration
    t1 = time.time()
    q_list = list()
    td_error = list()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        q = 0
        mse = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)  # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)
            r = r + 10
            ddpg.store_transition(s, a, r / 10, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995  # decay the action randomness
                q1, m1 = ddpg.learn()
                q_list.append(q1)
                td_error.append(m1)
                q += q1
                mse += m1
            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS - 1:
                aaa = 3
                # print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                if ep_reward > 1700: RENDER = True
                break
        print('Episode:', i, ' Reward: %i' % int(ep_reward), 'var: %.4f' % var,
              'q_value:= ' + str(q / MAX_EP_STEPS), 'td_error:= ' + str(mse / MAX_EP_STEPS))
        if ddpg.pointer > MEMORY_CAPACITY and i % 20 == 0:
            draw(q_list, td_error)
        if i % 50 == 0 and i > 0:
            torch.save(ddpg, "model/ddpg/Cat_Lstm_ddpg" + str(i) + ".pt")
    env.close()
    print('Running time: ', time.time() - t1)


# train()
