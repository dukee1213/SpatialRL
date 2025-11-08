from env_v1 import SpatialSpreadEnv
import random

env = SpatialSpreadEnv(render_mode = 'mp4')
env.reset()
na = env.num_agents
max_step = 1000

import numpy as np
import os
import pandas as pd
num_runs = 30
all_infection_rates = []
all_infected_counts = []
for run in range(num_runs):
    env = SpatialSpreadEnv(render_mode=None)
    obs, info = env.reset()
    run_infection_rates = []
    run_infected_counts = []
    
    for i in range(max_step):
        actions = {f"agent_{j}": random.randint(0, 3) for j in range(na)}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        infection_status = env.infection_status
        infected_total = np.sum(infection_status)
        infection_rate = infected_total / len(infection_status)

        run_infected_counts.append(infected_total)
        run_infection_rates.append(infection_rate)
        if any(terminations.values()) or all(truncations.values()):
            break
    env.close()
    final_step = i + 1
    if final_step < max_step:
        pad_len = max_step - final_step
        run_infected_counts += [run_infected_counts[-1]] * pad_len
        run_infection_rates += [run_infection_rates[-1]] * pad_len
    all_infected_counts.append(run_infected_counts)
    all_infection_rates.append(run_infection_rates)
avg_infected_counts = np.mean(all_infected_counts, axis=0)
avg_infection_rates = np.mean(all_infection_rates, axis=0)
std_dev_infection_rates = np.std(all_infection_rates, axis=0)
standard_error_infection_rates = std_dev_infection_rates / np.sqrt(num_runs)
one_sd = avg_infection_rates + standard_error_infection_rates
neg_one_sd = avg_infection_rates - standard_error_infection_rates
step_record = list(range(max_step))
df = pd.DataFrame({
    "step": step_record,
    "infection_rate": avg_infection_rates,
    "number_infected": avg_infected_counts,
    "standard_error": standard_error_infection_rates,
    "one_sd": one_sd,
    "neg_one_sd": neg_one_sd
})
df.to_excel("result/Fig1_No_Training_30Times.xlsx", index=False)
'''
for i in range(250):
    actions = {f"agent_{j}": random.randint(0, 3) for j in range(na)}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.render()
    if any(terminations.values()) or all(truncations.values()):
        break
env.close()
print(f"Simulation finished. Video is saved as './output.mp4'")
'''