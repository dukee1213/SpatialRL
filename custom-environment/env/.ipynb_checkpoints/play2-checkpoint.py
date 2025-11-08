from env_v1 import SpatialSpreadEnv
import random

env = SpatialSpreadEnv(render_mode = 'mp4')
env.reset()
na = env.num_agents
max_step = 1000
for i in range(250):
    actions = {f"agent_{j}": random.randint(0, 3) for j in range(na)}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.render()
    if any(terminations.values()) or all(truncations.values()):
        break
env.close()
print(f"Simulation finished. Video is saved as './output.mp4'")
