import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
from matplotlib import pyplot as plt
from Env_normal_DDPG import Environment
import warnings

#####################  hyper parameters  ####################

warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore")
env = Environment()
s_dim = env.state_dim + 2
MAX_EPISODES = 500
MAX_EP_STEPS = env.train_length
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 1  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128
RENDER = False


class ANet(nn.Module):  # ae(s)=a
    def __init__(self, _s_dim):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(_s_dim, 50)
        self.fc1.weight.data.normal_(0, 0.01)  # initialization
        self.fc2 = nn.Linear(50, 20)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(20, 1)
        self.out.weight.data.normal_(0, 0.01)  # initialization

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = 10 * torch.tanh(self.out(x))
        actions = x
        return actions


class CNet(nn.Module):  # ae(s)=a
    def __init__(self, _s_dim):
        super(CNet, self).__init__()
        self.lstm_s = nn.LSTM(s_dim, 10, 2)
        self.fcs = nn.Linear(10, 30)
        self.fcs.weight.data.normal_(0, 0.1)  # initialization
        self.fca = nn.Linear(1, 30)
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


class DDPG(nn.Module):
    def __init__(self, s_dim=0):
        super(DDPG, self).__init__()
        self.s_dim = s_dim
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + 2), dtype=np.float32)
        self.pointer = 0
        self.Actor_eval = ANet(s_dim)
        self.Actor_target = ANet(s_dim)
        self.Critic_eval = CNet(s_dim)
        self.Critic_target = CNet(s_dim)
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
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + 1])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs.unsqueeze(1), a)
        loss_a = -torch.mean(q)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_).detach()
        q_ = self.Critic_target(bs_.unsqueeze(1), a_).detach()
        q_target = br + GAMMA * q_
        q_v = self.Critic_eval(bs.unsqueeze(1), ba)
        td_error = self.loss_td(q_target, q_v)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()
        return -loss_a.data.item(), td_error.data.item()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1


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


def train():
    ddpg = DDPG(s_dim)
    torch.nn.utils.clip_grad_norm_(ddpg.parameters(), 10)
    var = 8
    t1 = time.time()
    q_list = list()
    td_error = list()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        q = 0
        mse = 0
        for j in range(env.train_length):
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a.data.item(), var), -10, 10)
            if a is np.NaN or abs(a) > 1000:
                print("error action := ", a, ' state: ', s)
                a = ddpg.choose_action(s)
                a = var * torch.randn((1, 1)).squeeze().data.item() + a
                # sys.exit()
            s_, r, r2 = env.step(a)
            r2 = max(-1000, min(r, 1000))
            r = (r / (env.balance + env.hold * env.current_price + 1)) * 10
            # r *= 10
            ddpg.store_transition(s, a, r, s_)
            if j % 10000 == 0 and j > 0:
                print(j, " reward: ", r2, " action:= ", a)
            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .99995
                q1, m1 = ddpg.learn()
                q_list.append(q1)
                td_error.append(m1)
                q += q1
                mse += m1
            s = s_
            ep_reward += r2
            if j == MAX_EP_STEPS - 1:
                a = 8
                break
        print('Episode:', i, ' Reward: ', ep_reward.data.item(), 'var: %.4f' % var,
              'q_value:= ' + str(q / MAX_EP_STEPS), 'td_error:= ' + str(mse / MAX_EP_STEPS))
        if ddpg.pointer > MEMORY_CAPACITY and i % 50 == 0:
            draw(q_list, td_error)
        if ddpg.pointer > MEMORY_CAPACITY and i % 10 == 0:
            torch.save(ddpg, "model2/ddpg/AE_LSTM_ddpg_" + str(i) + ".pt")
    print('Running time: ', time.time() - t1)


# train()