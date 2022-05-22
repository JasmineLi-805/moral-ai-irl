import torch
from torch import nn
import numpy as np

class TorchLinearReward(nn.Module):
    def __init__(self, n_input, n_h1=400, n_h2=1):
        super(TorchLinearReward, self).__init__()
        self.fc1 = nn.Linear(in_features=n_input, out_features=n_h1, bias=True)
        self.fc2 = nn.Linear(in_features=n_h1, out_features=n_h2, bias=True)
        self.act = nn.ELU()

    def forward(self, x):
        print(f'input={x}')
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        return x

    def get_theta(self):
        return [self.fc1.weight.detach(), self.fc2.weight.detach()]

    def get_rewards(self, states):
        if type(states) == np.ndarray:
            states = torch.tensor(states, dtype=torch.float)
        with torch.no_grad():
            rewards = self.forward(states).detach()
        return rewards

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


# class RecurrentReward(nn.Module):
#     def __init__(self):
#         super().__init__()
