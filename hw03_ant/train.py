import pybullet_envs_gymnasium
from gymnasium import make
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import copy

random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)
# torch.autograd.set_detect_anomaly(True)

EPS = 0.2
GAMMA = 0.99
TAU = 0.002
CRITIC_LR = 5e-4
BEGIN_ACTOR_LR = 2e-3
END_ACTOR_LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512
ENV_NAME = "AntBulletEnv-v0"
TRANSITIONS = 1_000_000


def soft_update(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)


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


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1)).view(-1)


class TD3:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_2 = Critic(state_dim, action_dim).to(DEVICE)

        self.actor_optim = Adam(self.actor.parameters(), lr=BEGIN_ACTOR_LR)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=CRITIC_LR)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=CRITIC_LR)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.replay_buffer = deque(maxlen=20_000)

    def update(self, transition):
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > BATCH_SIZE * 16:
            # Sample batch
            transitions = [self.replay_buffer[random.randint(0, len(self.replay_buffer) - 1)] for _ in
                           range(BATCH_SIZE)]
            state, action, next_state, reward, done = zip(*transitions)
            state = torch.tensor(np.array(state), device=DEVICE, dtype=torch.float)
            action = torch.tensor(np.array(action), device=DEVICE, dtype=torch.float)
            next_state = torch.tensor(np.array(next_state), device=DEVICE, dtype=torch.float)
            reward = torch.tensor(np.array(reward), device=DEVICE, dtype=torch.float)
            done = torch.tensor(np.array(done), device=DEVICE, dtype=torch.float)

            self.critic_1_optim.zero_grad()
            self.critic_2_optim.zero_grad()
            self.actor_optim.zero_grad()

            # Update critics
            Q_1 = self.critic_1(state, action)
            Q_2 = self.critic_2(state, action)

            with torch.no_grad():
                target_action = self.target_actor(next_state)
                Q_T_1 = self.target_critic_1(next_state, target_action)
                Q_T_2 = self.target_critic_2(next_state, target_action)
                Q_T = torch.min(Q_T_1, Q_T_2)

            critic_loss_1 = F.smooth_l1_loss(Q_1, reward + GAMMA * Q_T * torch.logical_not(done))
            critic_loss_2 = F.smooth_l1_loss(Q_2, reward + GAMMA * Q_T * torch.logical_not(done))

            critic_loss_1.backward(retain_graph=True)
            self.critic_1_optim.step()

            critic_loss_2.backward()
            self.critic_2_optim.step()

            # Update actor
            action_for_actor = self.actor(state)
            clip_action_for_actor = \
                torch.clip(action_for_actor + EPS * torch.empty(action.shape).normal_(mean=0, std=1), -1, +1)\
                .to(device=DEVICE, dtype=torch.float)
            Q_1_for_actor = self.critic_1(state, clip_action_for_actor)
            actor_loss = torch.mean(-Q_1_for_actor)

            actor_loss.backward()

            self.actor_optim.step()

            # Update target networks
            soft_update(self.target_critic_1, self.critic_1)
            soft_update(self.target_critic_2, self.critic_2)
            soft_update(self.target_actor, self.actor)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state), dtype=torch.float, device=DEVICE)
            return self.actor(state).cpu().numpy()

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
            state, reward, done, truncated, info = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make(ENV_NAME)
    test_env = make(ENV_NAME)
    td3 = TD3(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    state = env.reset()[0]
    episodes_sampled = 0
    steps_sampled = 0
    scheduler = CosineAnnealingLR(optimizer=td3.actor_optim, T_max=400_000, eta_min=END_ACTOR_LR)

    for i in range(TRANSITIONS):
        steps = 0

        # Epsilon-greedy policy
        action = td3.act(state)
        action = np.clip(action + EPS * np.random.randn(*action.shape), -1, +1)

        next_state, reward, done, truncated, info = env.step(action)
        done = done or truncated
        td3.update((state, action, next_state, reward, done))

        scheduler.step()

        state = next_state if not done else env.reset()[0]

        if (i + 1) % (TRANSITIONS // 1000) == 0:
            rewards = evaluate_policy(test_env, td3, 5)
            print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}, learning rate "
                  f"{td3.actor_optim.param_groups[0]['lr']}")
            td3.save()
