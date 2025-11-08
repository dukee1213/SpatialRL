from agilerl.algorithms.maddpg import MADDPG
import torch
import numpy as np
import os
from env.env_v1 import SpatialSpreadEnv
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckptpath", type=str, default="Jul12_I.pt", help="The model checkpoint path")

args = parser.parse_args()
checkpoint_path = os.path.join("models", "MADDPG", args.ckptpath)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = MADDPG.load(checkpoint_path, device)

env = SpatialSpreadEnv(render_mode="mp4")
max_steps = 1000

obs, info = env.reset()

for _ in range(max_steps):
    cont_actions, discrete_actions = agent.get_action(obs, training=False, infos=info)
    action = discrete_actions if agent.discrete_actions else cont_actions
    # action = {f'agent_{i}': np.array([np.random.randint(0, 4)]) for i in range(300)}
    obs, reward, terminations, truncations, info = env.step(action)
    env.render()
    if any(terminations.values()) or all(truncations.values()):
        break
env.close()