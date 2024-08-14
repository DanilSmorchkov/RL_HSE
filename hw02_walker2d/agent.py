import random
import numpy as np
import os
import torch
from torch import nn
from torch.nn import functional as F

from torch.distributions import Normal


def layer_init(layer, std=np.sqrt(2), bias_init=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_init)
    return layer


class model_NN_actor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(state_size, 256))
        self.fc2 = layer_init(nn.Linear(256, 256))
        self.fc3 = layer_init(nn.Linear(256, action_size), 0.01)
        self.tanh = nn.Tanh()

    def forward(self, state) -> torch.Tensor:
        x = self.tanh(self.fc1(state))
        x = self.tanh(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Advice: use same log_sigma for all states to improve stability
        # You can do this by defining log_sigma as nn.Parameter(torch.zeros(...))
        self.model = model_NN_actor(state_dim, action_dim)
        self.log_sigma = nn.Parameter(torch.zeros(action_dim))

    def compute_proba(self, state, action):
        # Returns probability of action according to current policy and distribution of actions
        mu = self.model(state)
        sigma = torch.exp(self.log_sigma)
        distribution = Normal(mu, sigma)
        return torch.exp(distribution.log_prob(action).sum(-1)), distribution

    def act(self, state):
        # Returns an action (with tanh), not-transformed action (without tanh) and distribution of non-transformed actions
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        with torch.no_grad():
            mu = self.model(state)
            sigma = torch.exp(self.log_sigma)
            distribution = Normal(mu, sigma)
            pure_action = distribution.sample()
            action = F.tanh(pure_action)
        return action, pure_action, distribution


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float()
            mu = self.model.model(state)
            sigma = torch.exp(self.model.log_sigma)
            distribution = torch.distributions.Normal(mu, sigma)
            pure_action = distribution.sample()
            action = torch.nn.functional.tanh(pure_action)
            return action

    def reset(self):
        pass

