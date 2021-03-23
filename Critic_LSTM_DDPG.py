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

warnings.filterwarnings("ignore")
MAX_EPISODES = 300
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
        self.lstm_s = nn.LSTM(s_dim, 10, 2)
        self.fcs = nn.Linear(10, 30)
        self.fcs.weight.data.normal_(0, 0.1)  # initialization
        self.fca = nn.Linear(_a_dim, 30)
        self.fca.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, _s, _a):
        x, _ = self.lstm_s(_s)
        x = x.view(-1, 10)
        x = self.fcs(x)
        y = self.fca(_a)
        net = F.relu(x + y)
        actions_value = self.out(net)
        return actions_value


class DDPG(object):
    def __init__(self, a_dim=0, s_dim=0, a_bound=0, ):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.Actor_eval = ANet(s_dim, a_dim)
        self.Actor_target = ANet(s_dim, a_dim)
        self.Critic_eval = CNet(s_dim, a_dim)
        self.Critic_target = CNet(s_dim, a_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.Actor_eval(s)[0].detach()

    def learn(self):
        for target_param, param in zip(self.Actor_target.parameters(), self.Actor_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
        for target_param, param in zip(self.Critic_target.parameters(), self.Critic_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs.unsqueeze(1), a)  # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
        # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
        loss_a = -torch.mean(q)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_).detach()  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(bs_.unsqueeze(1), a_).detach()  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = br + GAMMA * q_  # q_target = 负的
        # print(q_target)
        q_v = self.Critic_eval(bs.unsqueeze(1), ba)
        # print(q_v)
        td_error = self.loss_td(q_target, q_v)
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        # print(td_error)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()
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
    plt.title(u"LSTM-DDPG算法中Q_value随迭代次数变化曲线", fontproperties='SimHei')
    plt.xlabel(u"迭代次数", fontproperties='SimHei')
    plt.ylabel(u"Q-value信息", fontproperties='SimHei')
    plt.plot(q)
    plt.show()
    plt.figure(2)
    plt.title(u"LSTM-DDPG算法中TD-error随迭代次数变化曲线", fontproperties='SimHei')
    plt.xlabel(u"迭代次数", fontproperties='SimHei')
    plt.ylabel(u"TD-error信息", fontproperties='SimHei')
    plt.plot(td)
    plt.show()


# ddpg = DDPG(a_dim, s_dim)
#
# var = 3  # control exploration
# t1 = time.time()
# q_list = list()
# td_error = list()
# for i in range(MAX_EPISODES):
#     s = env.reset()
#     ep_reward = 0
#     q = 0
#     mse = 0
#     for j in range(MAX_EP_STEPS):
#         if RENDER:
#             env.render()
#
#         # s = torch.ra
#         # s = var*torch.randn((1, 3))+s
#         a = ddpg.choose_action(s)
#         a = np.clip(np.random.normal(a, var), -2, 2)  # add randomness to action selection for exploration
#         s_, r, done, info = env.step(a)
#         r = r + 10
#         ddpg.store_transition(s, a, r / 10, s_)
#
#         if ddpg.pointer > MEMORY_CAPACITY:
#             var *= .9995  # decay the action randomness
#             q1, m1 = ddpg.learn()
#             q_list.append(q1)
#             td_error.append(m1)
#             q += q1
#             mse += m1
#         s = s_
#         ep_reward += r
#         if j == MAX_EP_STEPS - 1:
#             aaa = 3
#             # print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
#             if ep_reward > 1700: RENDER = True
#             break
#     print('Episode:', i, ' Reward: %i' % int(ep_reward), 'var: %.4f' % var,
#           'q_value:= ' + str(q / MAX_EP_STEPS), 'td_error:= ' + str(mse / MAX_EP_STEPS))
#     if ddpg.pointer > MEMORY_CAPACITY and i % 20 == 0 and i > 0:
#         draw(q_list, td_error)
#         torch.save(ddpg, "model/ddpg/Critic_Lstm_ddpg_" + str(i) + ".pt")
# print('Running time: ', time.time() - t1)
