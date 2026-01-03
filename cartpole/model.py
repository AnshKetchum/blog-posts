import torch 
import torch.nn as nn 
import torch.nn.functional as F


class DeepQNetwork(nn.Module):
    def __init__(self, in_features, out_features, hidden_dimension = 512):
        super(DeepQNetwork, self).__init__()

        self.hidden_dimension = hidden_dimension
        self.in_features = in_features
        self.out_features = out_features

        self.fcn = nn.Sequential(
            nn.Linear(in_features, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, out_features)
        )

    def forward(self, states):
        o = self.fcn(states)
        return o

