import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.autograd import Variable
from matplotlib import pyplot as plt
from tqdm import tqdm
from Trade_Enviroment import Environment
from OU_process import OrnsteinUhlenbeckActionNoise as OU_process


#####################  hyper parameters  ####################

MAX_EPISODES = 300
LR_A = 0.001    # learning rate for actor
LR_C = 0.01    # learning rate for critic
GAMMA = 1    # reward discount
TAU = 0.001      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128
RENDER = False


def weights_init(m):
    nn.init.uniform_(m.weight.data, -0.003, 0.003)


class ANet(nn.Module):

    def __init__(self, _s_dim, _a_dim):
        super(ANet, self).__init__()
        self.s_dim = _s_dim
        self.norm1_1 = nn.BatchNorm1d(_s_dim-3)
        self.fc1_1 = nn.Linear(_s_dim-3, 100)
        self.fc1_1.apply(weights_init)
        # self.fc1_1.weight.data.uniform_(-0.3, 0.3)  # initialization
        self.norm1_2 = nn.BatchNorm1d(3)
        self.fc1_2 = nn.Linear(3, 100)
        self.fc1_2.apply(weights_init)
        self.norm2 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 50)
        self.fc2.apply(weights_init)
        self.norm3 = nn.BatchNorm1d(50)
        self.out = nn.Linear(50, _a_dim)
        self.out.apply(weights_init)  # initialization

    def forward(self, x, train_model=True):
        x1, x2 = x.split([self.s_dim-3, 3], dim=1)
        if train_model:
            x1 = self.norm1_1(x1)
            x2 = self.norm1_2(x2)
        x1 = torch.tanh(self.fc1_1(x1))
        # print(x1.data)
        x2 = torch.tanh(self.fc1_2(x2))
        x = (x1+x2)
        if train_model:
            x = self.norm2(x)
        x = torch.tanh(self.fc2(x))
        if train_model:
            x = self.norm3(x)
        x = self.out(x)
        x = torch.tanh(x)
        actions_value = x*1
        return actions_value


class CNet(nn.Module):
    def __init__(self, _s_dim, _a_dim):
        super(CNet, self).__init__()
        self.s_dim = _s_dim

        self.norm1_1 = nn.BatchNorm1d(_s_dim-3)
        self.fcs1_1 = nn.Linear(_s_dim-3, 100)
        self.fcs1_1.apply(weights_init)  # normal_(0, 0.001)  # initialization
        self.norm1_2 = nn.BatchNorm1d(3)
        self.fcs1_2 = nn.Linear(3, 100)
        self.fcs1_2.apply(weights_init)

        self.norm2 = nn.BatchNorm1d(100)
        self.fcs_2 = nn.Linear(100, 50)
        self.fcs_2.apply(weights_init)

        self.fca = nn.Linear(_a_dim, 50)
        self.fca.apply(weights_init)
        self.out_1 = nn.Linear(100, 50)
        self.out_1.apply(weights_init)
        self.out_2 = nn.Linear(50, 1)
        self.out_2.apply(weights_init)  # initialization

    def forward(self, _s, _a, train_model=True):
        x1, x2 = _s.split([self.s_dim - 3, 3], dim=1)

        if train_model:
            x1 = self.norm1_1(x1)
            x2 = self.norm1_2(x2)

        x1 = self.fcs1_1(x1)
        x2 = self.fcs1_2(x2)
        x = torch.tanh(x1+x2)

        if train_model:
            x = self.norm2(x)
        x = torch.tanh(self.fcs_2(x))
        y = torch.tanh(self.fca(_a))
        net = torch.cat((x, y), dim=1)
        net = torch.tanh(self.out_1(net))
        actions_value = self.out_2(net)
        # actions_value = net
        return actions_value


class DDPG(nn.Module):
    def __init__(self, _a_dim=0, _s_dim=0):
        super(DDPG, self).__init__()
        self.a_dim, self.s_dim = _a_dim, _s_dim
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.Actor_eval = ANet(_s_dim, _a_dim)
        self.Actor_target = ANet(_s_dim, _a_dim)
        self.Critic_eval = CNet(_s_dim, _a_dim)
        self.Critic_target = CNet(_s_dim, _a_dim)
        self.critic_train = torch.optim.Adam(self.Critic_eval.parameters(), lr=LR_C, weight_decay=0.02)
        self.actor_train = torch.optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.Actor_eval(s, train_model=False)[0].detach()

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

        # print(bs.data)
        miu_s_i = self.Actor_eval(bs)
        # print(a.data)
        q = self.Critic_eval(bs, miu_s_i)
        # print(q)
        # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
        loss_a = -torch.mean(q)
        self.actor_train.zero_grad()
        loss_a.backward()
        self.actor_train.step()
        # miu_s_i_2 = self.Actor_eval(bs).detach()
        # temp = self.Critic_eval(bs, miu_s_i_2).detach()
        # print("q value change: "+str((torch.mean(temp)+loss_a.data).data.item()))

        a_ = self.Actor_target(bs_)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(bs_, a_)   # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        y_i = br+GAMMA*q_
        q_v = self.Critic_eval(bs, ba)
        td_error = nn.MSELoss()(y_i, q_v)
        self.critic_train.zero_grad()
        td_error.backward()
        self.critic_train.step()
        # a__ = self.Actor_target(bs_).detach()
        # q__ = self.Critic_target(bs_, a__).detach()
        # y__i = br+GAMMA*q__
        # q__v = self.Critic_eval(bs, ba).detach()
        # td_error2 = nn.MSELoss()(q__v, y__i)
        # print("td_error change:= "+str((td_error-td_error2).detach()))
        return -loss_a.data.item(), td_error.data.item()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1


env = Environment()
s_dim = env.env_dim+3
a_dim = 1
MAX_EP_STEPS = env.train_length
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def draw(_q, _td):
    plt.close(plt.figure(1))
    plt.suptitle("q_value && td_error")
    q = np.array(_q)
    td = np.array(_td)
    plt1 = plt.subplot(2, 1, 1)
    plt1.set_title("q_value")
    plt.plot(q)
    plt2 = plt.subplot(2, 1, 2)
    plt2.set_title("td_error")
    plt.plot(td)
    # plt.cla()
    plt.show()


ddpg = DDPG(a_dim, s_dim)  # .to(device)
OU_process = OU_process()

def train(model_index):
    ddpg = DDPG(a_dim, s_dim)
    var = 0.5  # control exploration
    t1 = time.time()
    q_list = list()
    td_error = list()

    for i in tqdm(range(MAX_EPISODES)):
        s = env.reset()
        ep_reward = 0
        q = 0
        mse = 0
        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .95
        ave = 0
        for j in (range(MAX_EP_STEPS)):
            # s = torch.tensor(s, dtype=torch.float32).to(device)
            a = ddpg.choose_action(s)
            a = torch.clamp(torch.normal(a, var), -1, 1)
            # num = a.data.item()
            # if num < 0:
            #     _action = -1 * round(abs(num) * 10)
            # else:
            #     _action = round(num * 10)
            # print(num, "  :->  ", _action)
            # ave += _action
            # a = torch.clamp(a, -10, 10)    # add randomness to action selection for exploration
            # a = OU_process.count(x_pre=a)
            s_, r = env.step(a)
            # r = r+10
            r = torch.tensor([r], dtype=torch.float32)
            ddpg.store_transition(s, a, r/100, s_)
            if ddpg.pointer > MEMORY_CAPACITY:
                # var *= .9995    # decay the action randomness
                # print(a_dim)
                q1, m1 = ddpg.learn()
                q_list.append(q1)
                td_error.append(m1)
                q += q1
                mse += m1
            s = s_
            ep_reward += r
            # if j == MAX_EP_STEPS-1:
            #     # print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            #     break
        # print(ave)
        if i % 5 == 0:
            print(' Reward: %i' % int(ep_reward), 'var: %.4f' % var,
                'q_value:= ' + str(q / MAX_EP_STEPS), 'td_error:= ' + str(mse / MAX_EP_STEPS))
        if ddpg.pointer > MEMORY_CAPACITY and i % 10 == 0:
            draw(q_list, td_error)
        if i % 50 == 0:
            torch.save(ddpg, "model/ddpg_"+env.name+"_raw"+str(model_index)+".pt")
    print('Running time: ', time.time() - t1)


for i in range(10):
    train(i)
