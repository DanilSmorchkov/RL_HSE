import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        
    def act(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float)
        return self.model(state).cpu().numpy()

    def reset(self):
        pass

