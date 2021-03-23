import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_state, n_action, hidden_dim):
        super(Net, self).__init__()
        self.norm = nn.BatchNorm1d(n_state)
        self.fc1 = nn.Linear(n_state, hidden_dim)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(hidden_dim, n_action)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x, train=True):
        if train:
            x = self.norm(x)
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DuelingNet(nn.Module):
    def __init__(self, n_state, n_action, hidden_dim):
        super(DuelingNet, self).__init__()
        self.norm = nn.BatchNorm1d(n_state)
        self.fc1 = nn.Linear(n_state, hidden_dim)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc_V = nn.Linear(hidden_dim, 1)
        self.fc_V.weight.data.normal_(0, 0.1)
        self.fc_A = nn.Linear(hidden_dim, n_action)
        self.fc_A.weight.data.normal_(0, 0.1)

    def forward(self, x, train=True):
        if train:
            x = self.norm(x)
        x = self.fc1(x)
        x = F.relu(x)
        state_value = self.fc_V(x)
        adv_value = self.fc_A(x)
        actions_value = state_value + (adv_value-adv_value.mean())
        return actions_value
