import pybullet_envs_gymnasium
# Don't forget to install PyBullet!
import gymnasium
from gymnasium import make
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim import Adam

from torch.optim.lr_scheduler import CosineAnnealingLR
import random

random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)

ENV_NAME = "Walker2DBulletEnv-v0"

CONTINUE_TRAINING = False

LAMBDA = 0.95
GAMMA = 0.99

ACTOR_LR = 1e-3
END_ACTOR_LR = 5e-5
CRITIC_LR = 1e-4

CLIP = 0.2
VALUE_CLIP = 0.25
START_ENTROPY_COEF = 2e-2
END_ENTROPY_COEF = 1e-3
BATCHES_PER_UPDATE = 100
BATCH_SIZE = 256

MIN_TRANSITIONS_PER_UPDATE = 5000
MIN_EPISODES_PER_UPDATE = 5

ITERATIONS = 3000


def compute_lambda_returns_and_gae(trajectory):
    lambda_returns = []
    gae = []
    last_lr = 0.
    last_v = 0.

    for _, _, r, _, v in reversed(trajectory):
        ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
        last_lr = ret
        last_v = v
        lambda_returns.append(last_lr)
        gae.append(last_lr - v)

    # Each transition contains state, action, old action probability, value estimation and advantage estimation
    return [(s, a, p, v, adv) for (s, a, _, p, _), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]


def layer_init(layer, std=np.sqrt(2), bias_init=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_init)
    return layer


class Actor_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prob_action, old_prob_action, advantage, entropy):
        rel = prob_action / old_prob_action
        clipped = torch.clip(rel, 1 - CLIP, 1 + CLIP)
        loss = (torch.min(clipped * advantage, rel * advantage) + ENTROPY_COEF * entropy)
        return -torch.mean(loss)


class Critic_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, value, estimate_value):
        return F.mse_loss(estimate_value, value)


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


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )

    def get_value(self, state):
        return self.model(state)


class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optim = Adam(self.actor.parameters(), ACTOR_LR, eps=1e-5)
        self.critic_optim = Adam(self.critic.parameters(), CRITIC_LR, eps=1e-5)
        self.actor_loss = Actor_Loss()
        self.critic_loss = Critic_Loss()

    def update(self, trajectories):
        transitions = [t for traj in trajectories for t in traj]  # Turn a list of trajectories into list of transitions
        state, action, old_prob, target_value, advantage = zip(*transitions)
        state = np.array(state)
        action = np.array(action)
        old_prob = np.array(old_prob)
        target_value = np.array(target_value)
        advantage = np.array(advantage)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(BATCHES_PER_UPDATE):
            idx = np.random.randint(0, len(transitions), BATCH_SIZE)  # Choose random batch
            s = torch.tensor(state[idx])
            a = torch.tensor(action[idx]).float()
            op = torch.tensor(old_prob[idx]).float()  # Probability of the action in state s.t. old policy
            v = torch.tensor(target_value[idx]).float()  # Estimated by lambda-returns
            adv = torch.tensor(advantage[idx]).float()  # Estimated by generalized advantage estimation

            # Update actor here
            action_prob, distribution = self.actor.compute_proba(s, a)
            actor_loss = self.actor_loss(action_prob, op, adv, distribution.entropy().mean(dim=1))

            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            self.actor_optim.step()

            # Update critic here
            estimate_value = self.critic.get_value(s).flatten()
            critic_loss = self.critic_loss(estimate_value, v)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            self.critic_optim.step()

    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            action, pure_action, distr = self.actor.act(state)
            prob = torch.exp(distr.log_prob(pure_action).sum(-1))
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], prob.cpu().item()

    def save(self):
        torch.save(self.actor, "agent.pkl")


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        truncated = False
        state = env.reset()[0]
        total_reward = 0.

        while not done and not truncated:
            state, reward, done, truncated, info = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)
    return returns


def sample_episode(env, agent):
    s = env.reset()[0]
    done = False
    truncated = False
    trajectory = []
    while not done and not truncated:
        a, pa, p = agent.act(s)
        v = agent.get_value(s)
        new_state, reward, done, truncated, info = env.step(a)
        trajectory.append((s, pa, reward, p, v))
        s = new_state
    return compute_lambda_returns_and_gae(trajectory)


if __name__ == "__main__":
    env = make(ENV_NAME)
    # Normalization and clipping rewards
    # env = gymnasium.wrappers.NormalizeReward(env)
    # env = gymnasium.wrappers.TransformReward(env, lambda rew: np.clip(rew, -2, 2))
    ppo = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    if CONTINUE_TRAINING:
        ppo.actor = torch.load(__file__[:-8] + "/agent.pkl")
    state = env.reset()[0]
    episodes_sampled = 0
    steps_sampled = 0
    scheduler_actor = CosineAnnealingLR(
        ppo.actor_optim,
        T_max=ITERATIONS,
        eta_min=END_ACTOR_LR
    )

    for i in range(ITERATIONS):
        trajectories = []
        steps_ctn = 0

        ENTROPY_COEF = START_ENTROPY_COEF - i / ITERATIONS * (START_ENTROPY_COEF - END_ENTROPY_COEF)

        while len(trajectories) < MIN_EPISODES_PER_UPDATE or steps_ctn < MIN_TRANSITIONS_PER_UPDATE:
            traj = sample_episode(env, ppo)
            steps_ctn += len(traj)
            trajectories.append(traj)
        episodes_sampled += len(trajectories)
        steps_sampled += steps_ctn

        ppo.update(trajectories)

        scheduler_actor.step()

        if (i + 1) % (ITERATIONS//200) == 0:
            rewards = evaluate_policy(env, ppo, 5)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}, Episodes: {episodes_sampled}, Steps: {steps_sampled}")
            print(f'Learning rate: {ppo.actor_optim.param_groups[0]['lr']}')
            print(f'Entropy coefficient: {ENTROPY_COEF}')
            ppo.save()
