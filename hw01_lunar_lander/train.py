from gymnasium import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import random
import copy
import math


GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
TAU = 0
EPS_START = 0.8
EPS_END = 0.05
EPS_DECAY = 20000
SEED = 123


class model_NN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.15):
        super(model_NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

    def forward(self, state) -> torch.Tensor:
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class DQN:
    def __init__(self, state_dim, action_dim, maxlen=5000, batch_size=BATCH_SIZE, lr=LEARNING_RATE, gamma=GAMMA,
                 tau=TAU):
        self.steps = 0  # Do not change

        self.model = model_NN(state_dim, action_dim)   # Torch model
        # self.model = nn.Sequential(
        #     nn.Linear(state_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, action_dim)
        # )
        self.target_model = copy.deepcopy(self.model)

        self.buffer = deque(maxlen=maxlen)
        self.batch_size = batch_size

        self.lr = lr
        self.gamma = gamma
        self.tau = tau

        self.criterion = nn.HuberLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.buffer.append(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        batch = random.sample(self.buffer, self.batch_size)
        state, action, next_state, reward, done = list(zip(*batch))
        state_batch = torch.tensor(np.array(state), dtype=torch.float32)
        next_state_batch = torch.tensor(np.array(next_state), dtype=torch.float32)
        action_batch = torch.tensor(np.array(action), dtype=torch.int64).view(-1, 1)
        reward_batch = torch.tensor(np.array(reward), dtype=torch.float32)
        done_batch = torch.tensor(np.array(done), dtype=torch.int)
        return state_batch, action_batch, next_state_batch, reward_batch, done_batch
        
    def train_step(self, batch):
        # Use batch to update DQN's network.
        # Распакуем батч
        if self.batch_size > len(self.buffer):
            return
        state, action, next_state, reward, done = batch

        # Найдем значения Q-функции
        Q = self.model(state).gather(1, action)
        with torch.no_grad():
            Q_next = self.target_model(next_state).max(1).values
        Q_target = Q_next * self.gamma * torch.logical_not(done) + reward

        # Вычислим лосс
        loss = self.criterion(Q, Q_target.unsqueeze(1))

        # Прокинем градиенты
        self.optimizer.zero_grad()
        loss.backward()

        # Обновим модель, при этом ограничив значение градиентов
        nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or 
        # assign a values of network parameters via PyTorch methods.
        target_model_state_dict = self.target_model.state_dict()
        model_state_dict = self.model.state_dict()
        for key in target_model_state_dict:
            target_model_state_dict[key] = (target_model_state_dict[key] * self.tau +
                                            (1 - self.tau) * model_state_dict[key])
        self.target_model.load_state_dict(target_model_state_dict)

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = torch.tensor(np.array(state))
        with torch.no_grad():
            action = self.model(state)
        return np.argmax(np.array(action))

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model.state_dict(), "agent.pkl")


def evaluate_policy(agent, episodes=5, plot=False):
    if plot:
        env = make("LunarLander-v2", render_mode='human')
    else:
        env = make("LunarLander-v2")
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
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = EPS_START
    state = env.reset()[0]

    # Set random seeds
    # env.seed(SEED)
    # torch.manual_seed(SEED)
    # random.seed(SEED)
    # np.random.seed(SEED)
    # env.action_space.seed(SEED)
    
    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, truncated, info = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done or truncated))
        
        state = next_state if not done and not truncated else env.reset()[0]

    for i in range(TRANSITIONS):

        # Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * i / EPS_DECAY)

        next_state, reward, done, truncated, info = env.step(action)
        dqn.update((state, action, next_state, reward, done or truncated))

        state = next_state if not done and not truncated else env.reset()[0]

        if (i + 1) % (TRANSITIONS//100) == 0:
            print('Before evaluate policy')
            rewards = evaluate_policy(dqn, 30)
            print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            dqn.save()
