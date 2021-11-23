from torch import nn
import numpy as np

class TorchLinearReward(nn.Module):
    def __init__(self, num_in_feature):
        super().__init__()
        self.fc = nn.Linear(in_features=num_in_feature, out_features=1, bias=True)
        self.act = nn.ReLU6()

    def forward(self, x):
        print('in forward')
        x = self.fc(x)
        print('after fc')
        x = self.act(x)
        print('after act')
        return x

class LinearReward:
    def __init__(self, num_in_feature) -> None:
        self.num_in_feature = num_in_feature
        self.weights = np.random.rand(self.num_in_feature, 1)

    def updateWeights(self, weights):
        self.weights = weights
    
    def getRewards(self, states):
        assert states.shape[-1] == self.num_in_feature
        reward = np.matmul(states, self.weights)
        assert len(states.shape) == 1 or reward.shape[0] == states.shape[0]
        return reward



# class RecurrentReward(nn.Module):
#     def __init__(self):
#         super().__init__()
