from torch import nn

class LinearReward(nn.Module):
    def __init__(self, num_in_feature):
        super().__init__()
        self.fc = nn.Linear(in_features=num_in_feature, out_features=1, bias=True)
        self.act = nn.ReLU6()

    def forward(self, x):
        x = self.fc(x)
        return self.act(x)



# class RecurrentReward(nn.Module):
#     def __init__(self):
#         super().__init__()
