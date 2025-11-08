from agilerl.algorithms.maddpg import MADDPG
import torch
import numpy as np
import os
from env.env_v1 import SpatialSpreadEnv
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckptpath", type=str, default="MADDPG_trained_agent.pt", help="The model checkpoint path")
parser.add_argument("--social", type=bool, default=True, help="Whether to print social rewards")
args = parser.parse_args()
checkpoint_path = os.path.join("models", "MADDPG", args.ckptpath)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = MADDPG.load(checkpoint_path, device)

env = SpatialSpreadEnv(render_mode="mp4")
max_steps = 1000

obs, info = env.reset()
risk0_infected = []
risk1_infected = []
risk2_infected = []

for _ in range(max_steps):
    cont_actions, discrete_actions = agent.get_action(obs, training=False, infos=info)
    action = discrete_actions if agent.discrete_actions else cont_actions
    obs, reward, terminations, truncations, info = env.step(action)
    env.render()
    
    risk_types = env.risk_types
    infection_status = env.infection_status
    risk0_infected.append(np.sum(infection_status[risk_types == 0]))
    risk1_infected.append(np.sum(infection_status[risk_types == 1]))
    risk2_infected.append(np.sum(infection_status[risk_types == 2]))
    if any(terminations.values()) or all(truncations.values()):
        break
if args.social:
    with open("Social_Rewards.txt", "w") as f:
        for i in range(len(env.social)):
            f.write(f"Agent_{i}: {env.social[i]}\n")
    with open("Total_Rewards.txt", "w") as f:
        for agent_id, reward in env.accumR.items():
            f.write(f"{agent_id}: {reward}\n")
    with open("E_Rewards.txt", "w") as f:
        for i in range(len(env.Ecost)):
            f.write(f"Agent_{i}: {env.Ecost[i]}\n")
else:
    with open("Total_Rewards.txt", "w") as f:
        for agent_id, reward in env.accumR.items():
            f.write(f"{agent_id}: {reward}\n")
env.close()

total_infected = np.array(risk0_infected) + np.array(risk1_infected) + np.array(risk2_infected)
overall_rate = total_infected / 300
plt.figure(figsize=(10, 6))
plt.plot(overall_rate, label="Overall", color="red", linestyle="--")
plt.xlabel("Timestep")
plt.ylabel("Infection Rate")
plt.title("Infection Rate by Risk Type and Overall")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./infection_rate_plot_total.png")

risk0_rate = np.array(risk0_infected) / 100
risk1_rate = np.array(risk1_infected) / 100
risk2_rate = np.array(risk2_infected) / 100
plt.figure(figsize=(10, 6))
plt.plot(risk0_rate, label="Risk Averse", color="blue")
plt.plot(risk1_rate, label="Risk Seek", color="orange")
plt.plot(risk2_rate, label="Risk Neutral", color="green")
plt.xlabel("Timestep")
plt.ylabel("Infection Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./infection_rate_plot_groups.png")