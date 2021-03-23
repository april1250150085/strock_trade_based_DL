import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
from matplotlib import pyplot as plt
from Env_normal_DDPG import Environment
import warnings
import pandas as pd
from lstm_reg_simple import Net as Net


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
# regdata = pd.read_csv("D:/dataset/regdata/" + env.name + ".csv")
# reg_model = torch.load('D:/hjl_python_code/My_own_ddpg/model4/lstm_reg/lstm_reg_simple2.pt')

# ENV_NAME = 'Pendulum-v0'


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
        self.fcs = nn.Linear(_s_dim, 50)
        self.fcs.weight.data.normal_(0, 0.01)  # initialization
        self.fca = nn.Linear(1, 50)
        self.fca.weight.data.normal_(0, 0.01)
        self.out = nn.Linear(50, 1)
        self.out.weight.data.normal_(0, 0.01)  # initialization

    def forward(self, _s, _a):
        x = self.fcs(_s)
        y = self.fca(_a)
        net = F.relu(x + y)
        actions_value = self.out(net)
        return actions_value


class DDPG(nn.Module):
    def __init__(self, s_dim=0):
        super(DDPG, self).__init__()
        self.s_dim = s_dim
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + 1 + 1), dtype=np.float32)
        self.pointer = 0
        # self.sess = tf.Session()
        self.Actor_eval = ANet(s_dim)
        self.Actor_target = ANet(s_dim)
        self.Critic_eval = CNet(s_dim)
        self.Critic_target = CNet(s_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()
        torch.nn.utils.clip_grad_norm_(self.Actor_eval.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(self.Critic_eval.parameters(), 10)

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
        q = self.Critic_eval(bs, a)
        loss_a = -torch.mean(q)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_).detach()
        q_ = self.Critic_target(bs_, a_).detach()
        q_target = br + GAMMA * q_
        q_v = self.Critic_eval(bs, ba)
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


def draw(_loss, _reward):
    # plt.suptitle("loss_info and reward_info")
    plt1 = plt.subplot(2, 1, 1)
    plt1.set_title("loss information")
    plt.plot(_loss)
    plt2 = plt.subplot(2, 1, 2)
    plt2.set_title("q_target information")
    plt.plot(_reward)
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
            s_, r, r2 = env.step(a)
            r2 = max(-1000, min(r, 1000))
            r = (r / (env.balance + env.hold * env.current_price + 1)) * 10
            b = regdata.values[j, :]
            x_ = torch.tensor(b).unsqueeze(0)
            x_ = x_.unsqueeze(2)
            x_ = torch.tensor(x_, dtype=torch.float32)
            if j < 10:
                ddpg.store_transition(s, a, r, s_)
            else:
                next_p = env.max_price - (env.max_price - env.min_price) * reg_model(x_).data.item()
                if not abs(next_p - env.last_price) > 8.418 and (next_p - env.last_price) < 0:
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
            torch.save(ddpg, "model2/ddpg/norm_filter_ddpg_" + str(i) + ".pt")
    print('Running time: ', time.time() - t1)


# train()
