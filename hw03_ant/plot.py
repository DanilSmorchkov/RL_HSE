import pybullet_envs_gymnasium
import random

import numpy as np
from gymnasium import make
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for i in range(episodes):
        done = False
        truncated = False
        state = env.reset(seed=i)[0]
        total_reward = 0.

        while not done and not truncated:
            state, reward, done, truncated, info = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.model(state)

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
            state = torch.tensor(np.array(state), dtype=torch.float)
            return self.model(state).cpu().numpy()


ENV_NAME = "AntBulletEnv-v0"

env = make(ENV_NAME, render_mode='human')
env.reset(seed=2024)

agent = Agent()

rewards = evaluate_policy(env=env, agent=agent, episodes=50)

print(f"Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
