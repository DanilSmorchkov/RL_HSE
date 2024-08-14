import random
import numpy as np
import torch
import torch.nn as nn


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


class Agent:
    def __init__(self):
        self.model = model_NN(8, 4)
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pkl"))

    def act(self, state):
        state = torch.tensor(np.array(state))
        with torch.no_grad():
            action = self.model(state)
        return np.argmax(np.array(action))
