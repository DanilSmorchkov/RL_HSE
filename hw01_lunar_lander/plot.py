from agent import Agent
from train import evaluate_policy
import numpy as np


agent = Agent()

rewards = evaluate_policy(agent, episodes=60, plot=True)

print(f"Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")