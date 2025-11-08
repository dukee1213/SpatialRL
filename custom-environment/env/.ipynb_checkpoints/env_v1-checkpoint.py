import functools
import random
import math
from copy import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box
from pettingzoo import ParallelEnv
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
try:
    from env.PolyUtility import PolyUtility # access outsides or inside
    from env.BloomFilter import BloomFilter
except:
    from PolyUtility import PolyUtility 
    from BloomFilter import BloomFilter

uu = PolyUtility()
u = np.vectorize(uu.calU)

class SpatialSpreadEnv(ParallelEnv):
    metadata={
        "name": "custom_environment_v0",
    }
    def __init__(self, render_mode=None, side_len:int = 232, numA:int = 300):
        self.timestep = None
        self.render_mode = render_mode
        self.possible_agents = [f"agent_{i}" for i in range(numA)]
        self.position = None
        self.carefulness = None
        self.mobility = None
        self.infection_status = None
        self.risk_types = None
        self.num_contact = None
        
        self.side_len = side_len
        self.fig, self.ax = plt.subplots()
        self.scatter = None
        self.accumR = None
        self.flag = None
        self.flag2 = None
        self.social = None
        self.Ecost = None
        self.bf = None
        self.uni_contact = None

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        
        self.position = np.random.randint(0, self.side_len, size=(self.num_agents, 2)) 
        self.risk_types = np.array([i % 3 for i in range(self.num_agents)])
        self.carefulness = np.where(self.risk_types == 0, np.random.uniform(0.5, 1, self.num_agents),  
                   np.where(self.risk_types == 1, np.random.uniform(0, 0.5, self.num_agents),  
                   np.random.rand(self.num_agents))) 
        self.mobility = np.random.rand(self.num_agents)
        self.infection_status = np.zeros(self.num_agents, dtype=bool)
        self.infection_status[0] = True ## 
        self.num_contact = np.zeros(self.num_agents, dtype=int) 
        self.accumR = {ag: 50 for ag in self.possible_agents}
        self.social = np.zeros(self.num_agents)
        self.Ecost = np.zeros(self.num_agents)
        self.flag = False
        self.flag2 = False
        self.bf = [BloomFilter() for _ in range(self.num_agents)]
        self.uni_contact = np.zeros(self.num_agents, dtype=int)

        observations = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = self._get_observation(i)[0].flatten()
        infos = {a: {} for a in self.agents}
        return observations, infos



    
    def step(self, actions):
        rewards = np.zeros(self.num_agents, dtype=float)
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        mov_thres = 0.2
        current_Ir = np.mean(self.infection_status) * 100
        if current_Ir > 50:
            self.flag = True
            self.flag2 = True
        elif current_Ir < 30:
            self.flag = False
        for i, agent in enumerate(self.agents):
            action = actions[agent]
            carefulness_action = action // 2 
            mobility_action = action % 2 
            if carefulness_action == 0:
                self.carefulness[i] = min(1.0, self.carefulness[i] + 0.02) 
            elif carefulness_action == 1:
                self.carefulness[i] = max(0.0, self.carefulness[i] - 0.02)
            if mobility_action == 0:
                self.mobility[i] = min(1.0, self.mobility[i] + 0.02)
            else:
                self.mobility[i] = max(0.0, self.mobility[i] - 0.02)
            ''' Isolation '''
            # if self.infection_status[i]:
                # self.mobility[i] = max(0.3, self.mobility[i] - 0.02)
            ''' REMOVED in train '''
            # if self.risk_types[i] == 0 and current_Ir >= 30:# self.flag2
                # self.mobility[i] = 0
            ld = self._local_density(i)
            norm_ld = min(ld/20, 1)
            if np.random.rand() < self.mobility[i]:
                if np.random.rand() < mov_thres:
                    self.position[i] = np.random.uniform(0, self.side_len, size=2)
                else:
                    dist = 3.0
                    direction = np.random.choice(4)
                    if direction == 0:
                        self.position[i, 0] = (self.position[i, 0] - dist) % self.side_len
                    elif direction == 1:
                        self.position[i, 0] = (self.position[i, 0] + dist) % self.side_len
                    elif direction == 2:
                        self.position[i, 1] = (self.position[i, 1] - dist) % self.side_len
                    else:
                        self.position[i, 1] = (self.position[i, 1] + dist) % self.side_len
            else:
                continue
            ''' E Cost '''
            alpha = 0.3
            E_thres = 0.3
            m = (self.mobility[i])**(1)
            c = (self.carefulness[i])**(1)
            E = (1-c) * m * math.sqrt(norm_ld)
            rewards[i] += self._convert_utility(alpha*max(-E_thres, E_thres-E), self.agents[i], self.risk_types[i])
            self.Ecost[i] += alpha*max(-E_thres, E_thres-E)
        if self.timestep > 301:
            truncations = {a: True for a in self.agents}
        observations = {agent: self._get_observation(i)[0].flatten() for i, agent in enumerate(self.agents)}
        infos = {a: {} for a in self.agents}
        
        social_RAW = self._rateToFactor(np.mean(self.infection_status) * 100)
        for i, agent in enumerate(self.agents):
            if self.uni_contact[i] >= 20:
                self.bf[i] = BloomFilter()
                self.uni_contact[i] = 0
            if observations[agent][0] > 0: 
                nearby_agents = self._get_observation(i)[1]
                selected_neighbor = np.random.choice(nearby_agents)
                neigh_agent = self.agents[selected_neighbor]
                rand_num = np.random.rand()
                if i < selected_neighbor and rand_num>min(0.95, (self.carefulness[i]+self.carefulness[selected_neighbor])/2): # avoid double counting
                    socialR_S = social_RAW
                    s1 = 0 if self.infection_status[selected_neighbor] else 1
                    s2 = 0 if self.infection_status[i] else 1
                    ''' social event or NOT '''
                    if not self.bf[i].might_contain(selected_neighbor) and self.social[i] <= 10 and self.Ecost[i]<=20:
                        rewards[i] += self._convert_utility(socialR_S * s1 * s2, self.agents[i], self.risk_types[i]) 
                        self.social[i] += socialR_S * s1 * s2
                        self.bf[i].add(selected_neighbor)
                        self.uni_contact[i] += 1
                    socialR_S = social_RAW
                    s1 = 0 if self.infection_status[i] else 1
                    s2 = 0 if self.infection_status[selected_neighbor] else 1
                    if not self.bf[selected_neighbor].might_contain(i) and self.social[selected_neighbor] <= 10 and self.Ecost[selected_neighbor]<=20:
                        rewards[selected_neighbor] += self._convert_utility(socialR_S * s1 * s2, self.agents[selected_neighbor], self.risk_types[selected_neighbor])
                        self.social[selected_neighbor] += socialR_S * s1 * s2
                        self.bf[selected_neighbor].add(i)
                        self.uni_contact[selected_neighbor] += 1
                    
                    self.num_contact[i] += 1
                    self.num_contact[selected_neighbor] += 1
                    if self.infection_status[i] or self.infection_status[selected_neighbor]:
                        base_rate = 1.0
                        min_rate  = 0.0
                        avg_care = (self.carefulness[i] + self.carefulness[selected_neighbor]) / 2
                        infect_prob = min_rate + (1 - avg_care) * (base_rate-min_rate)
                        if np.random.rand() < infect_prob:
                            if not self.infection_status[i]:
                                self.infection_status[i] = True 
                            if not self.infection_status[selected_neighbor]:
                                self.infection_status[selected_neighbor] = True
                elif i < selected_neighbor:
                    continue
        
        ''' DiseaseCOst '''
        diseaseCost_RAW = -0.2
        for i, agent in enumerate(self.agents):
            if self.infection_status[i]:
                diseaseCost = diseaseCost_RAW
                rewards[i] += self._convert_utility(diseaseCost, self.agents[i], self.risk_types[i])
        if current_Ir > 10:
            infected_indices = [i for i in range(self.num_agents) if self.infection_status[i]]
            num_to_treat = int(len(infected_indices) * 0.05)
            selected_for_treatment = np.random.choice(infected_indices, num_to_treat, replace=False)
            for i in selected_for_treatment:
                self.infection_status[i] = False
        ''' END '''
        rewards = {agent: rewards[i] for i, agent in enumerate(self.agents)}
        self.timestep += 1
        observations = {agent: self._get_observation(i)[0].flatten() for i, agent in enumerate(self.agents)}
        if any(terminations.values()) or all(truncations.values()):
            self.agents = []
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        if self.render_mode is None:
            return
        if self.scatter is None or self.timestep == 0:
            self.ax.clear()
            self.scatter = self.ax.scatter([], [], s=40, edgecolors='none')  
            self.ax.set_xlim(0, self.side_len)
            self.ax.set_ylim(0, self.side_len)
            self.ax.set_xticks(range(0, self.side_len + 1, int(self.side_len // 5)))
            self.ax.set_yticks(range(0, self.side_len + 1, int(self.side_len // 5)))
            self.ax.tick_params(axis='both', labelsize=14)
            if self.render_mode in ('human'):
                plt.ion()
                plt.show()
        colors = np.where(self.infection_status, 'orangered', 'cornflowerblue')
        self.scatter.set_offsets(self.position)
        self.scatter.set_facecolor(colors)
        self.scatter.set_sizes(np.full(self.num_agents, 40)) 
        collected_steps = [1, 10, 50, 150, 200, 300]
        if self.timestep in collected_steps:
            self.ax.set_xlabel(f"t = {self.timestep}", fontsize=16, fontstyle='italic')
            self.ax.set_ylabel("")  # remove infection label
            plt.savefig(f"Fig_I_{self.timestep}.png", dpi=300, bbox_inches='tight')
        if self.render_mode in ('human'):
            plt.pause(0.08)
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        #return MultiDiscrete([11] + [2, 100, 3, 2, 5] * 10)
        return Box(low=-1, high=100, shape=(43,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        #return MultiDiscrete([5, 2])
        return Discrete(4)
    
    '''Helper methods'''
    def _convert_utility(self, reward:int = 1, agent: str = 'agent' , risk_type:int = 0):
        if agent not in self.accumR:
            raise ValueError(f"Agent {agent} not found in accumulated rewards!")
        old_value = self.accumR[agent]
        new_value = old_value + reward
        if new_value > 100 or new_value < 0:
            return 0
        self.accumR[agent] = new_value
        return u(new_value, risk_type = risk_type) - u(old_value, risk_type = risk_type)
        
    def _get_observation(self, agent_idx):
        distances = np.linalg.norm(self.position[agent_idx] - self.position, axis=1)
        nearby_agents = np.where(distances <= 10)[0] # 10公尺是否適合?
        nearby_agents = nearby_agents[nearby_agents != agent_idx]
        nearby_agents_X = np.where(distances <= 38)[0]
        nearby_agents_X = nearby_agents_X[nearby_agents_X != agent_idx]
        infected_count = 0
        if len(nearby_agents_X) > 0:
            for j, neighbor_idx in enumerate(nearby_agents_X[:len(nearby_agents_X)]):
                if self.infection_status[neighbor_idx]:
                    infected_count += 1
        memory = np.zeros(43, dtype=float)
        memory[0] = len(nearby_agents)
        memory[1] = self.infection_status[agent_idx]
        for j, neighbor_idx in enumerate(nearby_agents[:10]):  # At most 10 neighbors
            start_idx = 3 + j * 4 # what if no agent??
            memory[start_idx] = True
            memory[start_idx + 1] = self.num_contact[neighbor_idx]
            memory[start_idx + 2] = self.risk_types[neighbor_idx]
            memory[start_idx + 3] = self.carefulness[neighbor_idx] >= 0.5
        for j in range(len(nearby_agents), 10):
            start_idx = 3 + j * 4
            memory[start_idx] = False
            memory[start_idx + 1] = -1
            memory[start_idx + 2] = -1
            memory[start_idx + 3] = False
        memory[2] = infected_count
        return memory.astype(np.float32), nearby_agents

    def _local_density(self, agent_idx):
        distances = np.linalg.norm(self.position[agent_idx] - self.position, axis=1)
        nearby_agents_X = np.where(distances <= 38)[0]
        nearby_agents_X = nearby_agents_X[nearby_agents_X != agent_idx]
        infected_count = 0
        if len(nearby_agents_X) > 0:
            for j, neighbor_idx in enumerate(nearby_agents_X[:len(nearby_agents_X)]):
                if self.infection_status[neighbor_idx]:
                    infected_count += 1
        return infected_count

    def _rateToFactor(self, rate):
        if rate <= 50:
            reward_mul = 0.3
        else:
            reward_mul = 0
        return reward_mul
    def _helperI(self, x):
        if x <= 25:
            return 10
        elif x >= 45:
            return 0
        else:
            return -0.5 * (x - 25) + 10