import torch
from torch import nn
import numpy as np

class TorchLinCombReward(nn.Module):
    def __init__(self, n_input):
        super(TorchLinCombReward, self).__init__()
        self.fc = nn.Linear(in_features=n_input, out_features=1, bias=True)

    def forward(self, x):
        x = self.fc(x)
        return x

    def get_theta(self):
        return [self.fc.weight.detach()]

    def get_rewards(self, states):
        if type(states) == np.ndarray:
            states = torch.tensor(states, dtype=torch.float)
        with torch.no_grad():
            rewards = self.forward(states).detach()
        return rewards

class TorchLinearReward(nn.Module):
    def __init__(self, n_input, n_h1=400, n_h2=1):
        super(TorchLinearReward, self).__init__()
        self.fc1 = nn.Linear(in_features=n_input, out_features=n_h1, bias=True)
        self.fc2 = nn.Linear(in_features=n_h1, out_features=n_h2, bias=True)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        # x = self.act(x)
        return x

    def get_theta(self):
        return [self.fc1.weight.detach(), self.fc2.weight.detach()]

    def get_rewards(self, states):
        if type(states) == np.ndarray or type(states) == tuple:
            states = torch.tensor(states, dtype=torch.float)
        with torch.no_grad():
            rewards = self.forward(states).detach()
        return rewards

class TorchRNNReward(nn.Module):
    def __init__(self, n_input, n_h1=400, n_h2=1):
        super(TorchRNNReward, self).__init__()
        self.fc1 = nn.Linear(in_features=n_input, out_features=n_h1 // 2, bias=True)
        self.rnn = nn.RNN(input_size=n_h1 // 2, hidden_size=n_h1, num_layers=1)
        self.fc2 = nn.Linear(in_features=n_h1, out_features=n_h2, bias=True)
        self.act = nn.ReLU6()

    def forward(self, x, h):
        x = self.fc1(x)
        x = self.act(x)
        if len(x.shape) == 1:
            x = x[None, None, :]
        elif len(x.shape) == 2:
            x = x[None, :]
        x, hid = self.rnn(x, h)
        x = x.squeeze()
        x = self.act(x)
        x = self.fc2(x)
        return x, hid

    def get_theta(self):
        return [self.fc1.weight.detach(), self.fc2.weight.detach()]

    def get_rewards(self, states, hidden):
        if type(states) == np.ndarray:
            states = torch.tensor(states, dtype=torch.float)
        with torch.no_grad():
            rewards, hid = self.forward(states, hidden)
            rewards = rewards.detach()
            hid = hid.detach()
        return rewards, hid

class LinearReward:
    def __init__(self, num_in_feature) -> None:
        self.num_in_feature = num_in_feature
        self.weights = np.random.rand(self.num_in_feature, 1)

    def updateWeights(self, weights):
        self.weights = weights
    
    def getRewards(self, states):
        assert states.shape[-1] == self.num_in_feature, f'input shape: {states.shape[-1]}, feat shape: {self.num_in_feature}'
        reward = np.matmul(states, self.weights)
        assert len(states.shape) == 1 or reward.shape[0] == states.shape[0]
        return reward
