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
MAX_EPISODES = 1000
MAX_EP_STEPS = 200
LR_A = 0.01  # learning rate for actor
LR_C = 0.02  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128
RENDER = False
ENV_NAME = 'Pendulum-v0'


###############################  DDPG  ####################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.net = nn.LSTM(3, 10, 2)
        self.fc = nn.Linear(3, 30)
        self.fc.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # x, _ = self.net(x)
        # x = x.view(-1, 10)
        # x = x.squeeze(0)
        x = self.fc(x)
        return x


net = Net()


class ANet(nn.Module):  # ae(s)=a
    def __init__(self, _s_dim, _a_dim):
        super(ANet, self).__init__()
        self.fc_1 = Net()
        self.fc_s = nn.Linear(_s_dim, 30)
        self.fc_s.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, _a_dim)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = torch.relu(net(x))
        # x = torch.relu(self.fc_1(x))
        # net.fc.weight.requires_grad = False
        # net.fc.bias.requires_grad = False
        x = self.out(x)
        x = torch.tanh(x)
        actions_value = x * 2
        return actions_value


class CNet(nn.Module):
    def __init__(self, _s_dim, _a_dim):
        super(CNet, self).__init__()
        # self.net = net()
        self.fc_1 = Net()
        self.fc_s = nn.Linear(_s_dim, 30)
        self.fc_s.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(_a_dim, 30)
        self.fca.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, _s, _a):
        # x = self.fc_1(_s)
        x = net(_s)
        # net.fc.weight.requires_grad = False
        # net.fc.bias.requires_grad = False
        y = self.fca(_a)
        net2 = torch.relu(x + y)
        actions_value = self.out(net2)
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
        # s = s.unsqueeze(0)
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
        q = self.Critic_eval(bs, a)
        loss_a = -torch.mean(q)
        self.Actor_eval.fc_1.fc.weight.requires_grad = False
        self.Actor_eval.fc_1.fc.bias.requires_grad = False
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()
        self.Actor_eval.fc_1.fc.weight.requires_grad = True
        self.Actor_eval.fc_1.fc.bias.requires_grad = True

        ba2 = self.Actor_eval(bs)
        q_temp = self.Critic_eval(bs, ba2)
        self.Actor_eval.out.weight.requires_grad = False
        self.Actor_eval.out.bias.requires_grad = False
        self.Critic_eval.fca.weight.requires_grad = False
        self.Critic_eval.fca.bias.requires_grad = False
        self.Critic_eval.out.weight.requires_grad = False
        self.Critic_eval.out.bias.requires_grad = False
        loss_temp = -2 * torch.mean(q_temp)
        self.atrain.zero_grad()
        loss_temp.backward()
        self.atrain.step()
        self.Actor_eval.out.weight.requires_grad = True
        self.Actor_eval.out.bias.requires_grad = True
        self.Critic_eval.fca.weight.requires_grad = True
        self.Critic_eval.fca.bias.requires_grad = True
        self.Critic_eval.out.weight.requires_grad = True
        self.Critic_eval.out.bias.requires_grad = True

        a_ = self.Actor_target(bs_).detach()
        q_ = self.Critic_target(bs_, a_).detach()
        q_target = br + GAMMA * q_
        q_v = self.Critic_eval(bs, ba)
        td_error = self.loss_td(q_target, q_v)
        self.Critic_eval.fc_1.fc.weight.requires_grad = False
        self.Critic_eval.fc_1.fc.bias.requires_grad = False
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()
        self.Critic_eval.fc_1.fc.weight.requires_grad = True
        self.Critic_eval.fc_1.fc.bias.requires_grad = True

        # loss_2 = torch.mean((self.Critic_eval(bs, ba)-q_target)*self.Critic_eval(bs, ba))
        # self.Actor_eval.out.weight.requires_grad = False
        # self.Actor_eval.out.bias.requires_grad = False
        # self.Critic_eval.fca.weight.requires_grad = False
        # self.Critic_eval.fca.bias.requires_grad = False
        # self.Critic_eval.out.weight.requires_grad = False
        # self.Critic_eval.out.bias.requires_grad = False
        # self.ctrain.zero_grad()
        # loss_2.backward()
        # self.ctrain.step()
        # self.Actor_eval.out.weight.requires_grad = True
        # self.Actor_eval.out.bias.requires_grad = True
        # self.Critic_eval.fca.weight.requires_grad = True
        # self.Critic_eval.fca.bias.requires_grad = True
        # self.Critic_eval.out.weight.requires_grad = True
        # self.Critic_eval.out.bias.requires_grad = True

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


def draw(_q, _td):
    q = np.array(_q)
    td = np.array(_td)
    plt.figure(1)
    plt.title(u"Share-DDPG算法中Q_value随迭代次数变化曲线", fontproperties='SimHei')
    plt.xlabel(u"迭代次数", fontproperties='SimHei')
    plt.ylabel(u"Q-value信息", fontproperties='SimHei')
    plt.plot(q)
    plt.show()
    plt.figure(2)
    plt.title(u"Share-DDPG算法中TD-error随迭代次数变化曲线", fontproperties='SimHei')
    plt.xlabel(u"迭代次数", fontproperties='SimHei')
    plt.ylabel(u"TD-error信息", fontproperties='SimHei')
    plt.plot(td)
    plt.show()


ddpg = DDPG(a_dim, s_dim)


def train():
    var = 3  # control exploration
    t1 = time.time()
    q_list = list()
    td_error = list()
    RENDER = False
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        q = 0
        mse = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # s = torch.tensor(var * torch.randn((1, 3)).squeeze(0) + s, dtype=torch.float32)
            # s = s.unsqueeze(0)
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)  # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)
            r = r + 10
            ddpg.store_transition(s, a, r / 10, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9999995  # decay the action randomness
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
                if ep_reward > 1500: RENDER = True
                break
        print('Episode:', i, ' Reward: %i' % int(ep_reward), 'var: %.4f' % var,
              'q_value:= ' + str(q / MAX_EP_STEPS), 'td_error:= ' + str(mse / MAX_EP_STEPS))
        if ddpg.pointer > MEMORY_CAPACITY and i % 20 == 0 and i > 0:
            draw(q_list, td_error)
            # torch.save(ddpg, "model/ddpg/share_Lstm_ddpg_" + str(i) + ".pt")
        if i % 50 == 0:
            torch.save(ddpg, "model/ddpg/share_Lstm_ddpg_" + str(i) + ".pt")
    env.close()
    print('Running time: ', time.time() - t1)


# train()
