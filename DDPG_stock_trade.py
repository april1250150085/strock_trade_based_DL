import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import warnings
from tqdm import tqdm
from torch.autograd import Variable
from matplotlib import pyplot as plt
from Trade_Enviroment import Environment
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")
#####################  hyper parameters  ####################

MAX_EPISODES = 300
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.99    # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128
RENDER = False
plt.figure(1)
###############################  DDPG  ####################################


class ANet(nn.Module):   # ae(s)=a
    def __init__(self, _s_dim, _a_dim):
        super(ANet, self).__init__()
        self.norm1 = nn.BatchNorm1d(_s_dim)
        self.fc1 = nn.Linear(_s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(30, _a_dim)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        # x = torch.Tensor(x)
        # x = self.norm1(x)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        x = torch.tanh(x)
        actions_value = x*10
        return actions_value


class CNet(nn.Module):   # ae(s)=a
    def __init__(self, _s_dim, _a_dim):
        super(CNet, self).__init__()
        self.norm_s = nn.BatchNorm1d(_s_dim)
        self.fcs = nn.Linear(_s_dim, 30)
        self.fcs.weight.data.normal_(0, 0.1)  # initialization
        self.norm_a = nn.BatchNorm1d(_a_dim)
        self.fca = nn.Linear(_a_dim, 30)
        self.fca.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, _s, _a):
        # _s = self.norm_s(_s)
        x = self.fcs(_s)
        # _a = self.norm_a(_a)
        y = self.fca(_a)
        net = F.relu(x+y)
        actions_value = self.out(net)
        return actions_value


class DDPG(nn.Module):
    def __init__(self, a_dim=0, s_dim=0, a_bound=0,):
        super(DDPG, self).__init__()
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.memory = torch.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=torch.float32).to(device)
        self.pointer = 0
        self.Actor_eval = ANet(s_dim, a_dim).to(device)
        self.Actor_target = ANet(s_dim, a_dim).to(device)
        self.Critic_eval = CNet(s_dim, a_dim).to(device)
        self.Critic_target = CNet(s_dim, a_dim).to(device)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        # s = torch.unsqueeze(, 0)
        s = s.unsqueeze(0)
        return self.Actor_eval(s)[0].detach()

    def learn(self):

        for target_param, param in zip(self.Actor_target.parameters(), self.Actor_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
        for target_param, param in zip(self.Critic_target.parameters(), self.Critic_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)


        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        indices = torch.tensor(indices, dtype=torch.long).to(device)
        # bt = self.memory[indices, :].detach().requires_grad(True)
        # bt = torch.tensor(self.memory[indices, :]).to(device)
        bt = self.memory[indices, :]
        # bs = torch.tensor(bt[:, :self.s_dim])
        # ba = torch.tensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        # br = torch.tensor(bt[:, -self.s_dim - 1: -self.s_dim])
        # bs_ = torch.tensor(bt[:, -self.s_dim:])

        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs, a)  # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
        # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
        loss_a = -torch.mean(q)
        #print(q)
        #print(loss_a)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_).detach()  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(bs_, a_).detach()   # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = br+GAMMA*q_  # q_target = 负的
        q_v = self.Critic_eval(bs, ba)
        td_error = self.loss_td(q_target, q_v)
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        #print(td_error)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()
        return -loss_a.data.item(), td_error.data.item()

    def store_transition(self, s, a, r, s_):
        temp = torch.cat((s, a), dim=0)
        temp = torch.cat((temp, r), dim=0)
        transition = torch.cat((temp, s_), dim=0)
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1


###############################  training  ####################################
env = Environment()
env = env.to(device)
s_dim = env.env_dim
a_dim = 1
MAX_EP_STEPS = env.train_length

# print(device)

def draw(_q, _td):
    plt.close(plt.figure(1))
    plt.suptitle("q_value && td_error")
    # q = np.array(_q)
    q = torch.tensor(_q)
    # td = np.array(_td)
    td = torch.tensor(_td)
    plt1 = plt.subplot(2, 1, 1)
    plt1.set_title("q_value")
    plt.plot(q)
    plt2 = plt.subplot(2, 1, 2)
    plt2.set_title("td_error")
    plt.plot(td)
    # plt.cla()
    plt.show()


def train(model_index):
    ddpg = DDPG(a_dim, s_dim)
    ddpg.to(device)
    var = 3  # control exploration
    t1 = time.time()
    q_list = list()
    td_error = list()
    for i in tqdm(range(MAX_EPISODES)):
        s = env.reset()
        ep_reward = 0
        q = 0
        mse = 0
        for j in (range(MAX_EP_STEPS)):
            # s = torch.tensor(s, dtype=torch.float32).to(device)
            a = ddpg.choose_action(s)
            a = torch.clamp(torch.normal(a, var), -10, 10)    # add randomness to action selection for exploration
            s_, r = env.step(a)
            # a = torch.tensor(a).to(device)
            # s_ = torch.tensor(s_, dtype=torch.float32).to(device)
            r = r+10
            r = torch.tensor([r], dtype=torch.float32).to(device)
            ddpg.store_transition(s, a, r / 100, s_)
            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                q1, m1 = ddpg.learn()
                q_list.append(q1)
                td_error.append(m1)
                q += q1
                mse += m1
            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS-1:
                # print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                break
        if i % 10 == 0:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'var: %.4f' % var,
                  'q_value:= ' + str(q / MAX_EP_STEPS), 'td_error:= ' + str(mse / MAX_EP_STEPS))
            if ddpg.pointer > MEMORY_CAPACITY and i % 10 == 0:
                draw(q_list, td_error)
    torch.save(ddpg, "model/ddpg_"+env.name+"_raw_temp_"+str(model_index)+".pt")
    print('Running time: ', time.time() - t1)


if __name__ == "__main__":
    # for i in range(1):
    train(0)
