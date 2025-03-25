import os
import gym
import json
import random
import imageio, imageio_ffmpeg
import numpy as np
import torch as th
from gym import spaces
import matplotlib.pyplot as plt

positions_directory = "/"
SimulationVariables = {
    "SimAgents" : 6,
    "AgentData" : [],
    "SafetyRadius" : 0.25,
    "AccelerationInit" : 0.0,
    "NeighborhoodRadius" : 2,
    "VelocityUpperLimit" : 2.5,
    "AccelerationUpperLimit" : 5.0,
    "X" : 2,
    "Y" : 2,
    "dt" : 0.1,
    "EvalTimeSteps" : 300,
    "LearningTimeSteps" : 600000,
    "Episodes" : 10,
    "LearningRate": 0.005,
    "NumEnvs": 6,
    "Counter" : 0,
    "Seed" : 23,
    "ModelSeed" : 19,
}

class Agent:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.acceleration = np.zeros(2)
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"] #See if this needs to be fixed
        self.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"], size=2), 2)
        self.max_velocity = SimulationVariables["VelocityUpperLimit"]


    def update(self, action):
        self.acceleration += action
        acc_magnitude = np.linalg.norm(self.acceleration)
        if acc_magnitude > 0:
            if acc_magnitude > SimulationVariables["AccelerationUpperLimit"]:
                scaled_magnitude = SimulationVariables["AccelerationUpperLimit"] * np.tanh(acc_magnitude / SimulationVariables["AccelerationUpperLimit"])
                self.acceleration = (self.acceleration / acc_magnitude) * scaled_magnitude
        self.velocity += self.acceleration * SimulationVariables["dt"]
        vel = np.linalg.norm(self.velocity)
        if vel > 0:
            if vel > self.max_velocity:
                self.velocity = self.velocity * np.tanh(self.max_velocity / vel)

        self.position += self.velocity * SimulationVariables["dt"]
        # self.heading = self.find_heading(velocity)

        return self.position, self.velocity

    # def find_heading(self, velocity):
    #     final_vel_magnitude = np.linalg.norm(velocity)
    #     if final_vel_magnitude > 1e-8:
    #         heading = self.velocity / final_vel_magnitude
    #     else:
    #         heading = np.array([0.0, 0.0], dtype=float)

    #     return heading

class FlockingEnv(gym.Env):
    def __init__(self):
        super(FlockingEnv, self).__init__()
        self.episode = 0
        self.counter = 11
        self.CTDE = True  # Enable centralized training
        self.current_timestep = 0
        self.num_agents = SimulationVariables["SimAgents"]

        self.reward_components = {
            'alignment': [],
            'cohesion_penalty': [],
            'separation_penalty': [],
            'diversity_bonus': [],
            'cohesion_bonus': [],
            'out_of_flock_penalty': [],
            'formation_penalty': [],
            'collsion_penalty': [],
            'total': []
        }

        # Multi-agent spaces
        self.agents = [Agent(position) for position in self.read_agent_locations()]
        self.observation_space = spaces.Dict({
            i: spaces.Box(low=-np.inf, high=np.inf, shape=(4 + 4*(self.num_agents-1),), dtype=np.float32)
            for i in range(self.num_agents)
        })
        self.action_space = spaces.Dict({
            i: spaces.Box(low=-5, high=5, shape=(2,), dtype=np.float32)
            for i in range(self.num_agents)
        })

        # Centralized storage for MADDPG
        self.centralized_observations = {i: None for i in range(self.num_agents)}
        self.reward_log = []
        self.cumulative_rewards = {i: 0 for i in range(self.num_agents)}

    def step(self, action_dict):
        self.current_timestep += 1
        actions = [action_dict[i] for i in range(self.num_agents)]
        observations = self.simulate_agents(actions)
        rewards, dones = self.calculate_reward()

        # Convert to MADDPG format
        obs_dict = {i: observations[i] for i in range(self.num_agents)}
        reward_dict = {i: rewards[i] for i in range(self.num_agents)}
        done_dict = {"__all__": any(dones.values())}  # Global termination

        return obs_dict, reward_dict, done_dict, {}

    def reset(self):
        self.agents = [Agent(position) for position in self.read_agent_locations()]
        for agent in self.agents:
            agent.velocity = np.random.uniform(-2.5, 2.5, size=2)

        # Get centralized observations for all agents
        self.centralized_observations = self.get_centralized_observations()
        return {i: self.get_agent_observation(i) for i in range(self.num_agents)}

    def test_reset_1(self):
        self.agents = [Agent([0,0.0]) for i in range(self.num_agents)]
        for agent in self.agents:
            agent.velocity = np.random.uniform(-2.5, 2.5, size=2)

        # Get centralized observations for all agents
        self.centralized_observations = self.get_centralized_observations()
        return {i: self.get_agent_observation(i) for i in range(self.num_agents)}

    def test_reset_2(self):
        pos = [[1.0,1] ,[2,1],[3,1],[1,0] , [2,0] ,[3,0]]
        vel = [[-1,1.0],[0,1],[1,1],[-1,-1],[0,-1],[1,-1]]
        self.agents = [Agent(p) for p in pos]
        for i, agent in enumerate(self.agents):
            agent.velocity = vel[i]

        # Get centralized observations for all agents
        self.centralized_observations = self.get_centralized_observations()
        return {i: self.get_agent_observation(i) for i in range(self.num_agents)}

    def close(self):
        print("Simulation is complete. Cleaned Up!.")

    def simulate_agents(self, actions):
        """Process actions for all agents"""
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            agent.update(action)
        return [self.get_agent_observation(i) for i in range(self.num_agents)]

    def check_collision(self, agent):
        for other in self.agents:
            if agent != other:
                distance = np.linalg.norm(agent.position - other.position)
                if distance < SimulationVariables["SafetyRadius"]/2:
                    return True
        return False

    def get_agent_observation(self, agent_id):
        """Individual observation with neighbor context"""
        agent = self.agents[agent_id]
        neighbor_pos, neighbor_vel = self.get_closest_neighbors(agent)

        # Calculate swarm centroid
        all_positions = np.array([a.position for a in self.agents])
        all_velocities = np.array([a.velocity for a in self.agents])
        centroid_pos = np.mean(all_positions, axis=0)
        centroid_vel = np.mean(all_velocities, axis=0)

        # Flatten neighbor info (pad if no neighbors)
        max_neighbors = self.num_agents - 1
        neighbor_pos = np.array(neighbor_pos).flatten()
        neighbor_vel = np.array(neighbor_vel).flatten()

        # Calculate how many neighbors are missing
        num_neighbors = len(neighbor_pos) // 2  # Each neighbor has 2D position
        missing_neighbors = max_neighbors - num_neighbors

        # Create centroid-based padding
        pad_pos = np.tile(centroid_pos, missing_neighbors).flatten()
        pad_vel = np.tile(centroid_vel, missing_neighbors).flatten()

        padded_neighbor_pos = np.concatenate([neighbor_pos, pad_pos])
        padded_neighbor_vel = np.concatenate([neighbor_vel, pad_vel])

        return np.concatenate([
            agent.position,           # 2D
            agent.velocity,           # 2D
            padded_neighbor_pos,      # 4 * (num_agents - 1)
            padded_neighbor_vel       # 4 * (num_agents - 1)
        ], dtype=np.float32)


    def get_centralized_observations(self):
        """For centralized critics during training"""
        return np.array([agent.position for agent in self.agents]).flatten()

    def get_closest_neighbors(self, agent):
        neighbor_positions=[]
        neighbor_velocities=[]

        for _, other in enumerate(self.agents):
            if agent != other:
                distance = np.linalg.norm(other.position - agent.position)

                if(distance <= SimulationVariables["NeighborhoodRadius"]):
                        neighbor_positions.append(other.position)
                        neighbor_velocities.append(other.velocity)

        return neighbor_positions, neighbor_velocities

    def calculate_reward(self):
        rewards = {}
        dones = {}
        component_store = {
            'alignment': [],
            'cohesion_penalty': [],
            'separation_penalty': [],
            'diversity_bonus': [],
            'cohesion_bonus': [],
            'out_of_flock_penalty': [],
            'formation_penalty': [],
            'collsion_penalty': [],
            'total': []
        }

        component_store['alignment'].append(0)
        component_store['cohesion_penalty'].append(0)
        component_store['separation_penalty'].append(0)
        component_store['diversity_bonus'].append(0)
        component_store['cohesion_bonus'].append(0)
        component_store['out_of_flock_penalty'].append(0)
        component_store['formation_penalty'].append(0)
        component_store['collsion_penalty'].append(0)
        component_store['total'].append(0)

        for i, agent in enumerate(self.agents):
            neighbor_pos, neighbor_vel = self.get_closest_neighbors(agent)
            (total, alignment, cohesion_pen, diversity, cohesion_bonus, separation,
             out_of_flock_penalty, formation_penalty, collsion_penalty, out_of_flock) = self.reward(agent, neighbor_vel, neighbor_pos)

            # Store components per agent
            component_store['alignment'][-1] += (alignment)
            component_store['cohesion_penalty'][-1] += (cohesion_pen)
            component_store['separation_penalty'][-1] += (separation)
            component_store['diversity_bonus'][-1] += (diversity)
            component_store['cohesion_bonus'][-1] += (cohesion_bonus)
            component_store['out_of_flock_penalty'][-1] += (out_of_flock_penalty)
            component_store['formation_penalty'][-1] += (formation_penalty)
            component_store['collsion_penalty'][-1] += (out_of_flock_penalty)
            component_store['total'][-1] += (total)

            # Existing termination logic
            dones[i] = self.check_collision(agent) or out_of_flock
            rewards[i] = total #- 50 if dones[i] else total

        # Aggregate and store per timestep
        for key in component_store:
            self.reward_components[key].append((component_store[key]))

        return rewards, dones

    # def compute(self, agent):
    #     neighbor_positions, neighbor_velocities = self.get_closest_neighbors(agent)
    #     center_of_mass = np.mean(neighbor_positions, axis=0)
    #     distance_to_com = np.linalg.norm(agent.position - center_of_mass)
    #     direction_to_com = center_of_mass - agent.position

    #     if np.linalg.norm(direction_to_com) > 1e-8:
    #         angle_to_com = np.arctan2(direction_to_com[1], direction_to_com[0])
    #     else:
    #         angle_to_com = 0.0

    #     agent_speed = np.linalg.norm(agent.velocity)
    #     neighbor_speeds = [np.linalg.norm(v) for v in neighbor_velocities]
    #     avg_neighbor_speed = np.mean(neighbor_speeds) if neighbor_speeds else 0.0
    #     speed_diff = agent_speed - avg_neighbor_speed

    #     return angle_to_com, distance_to_com, speed_diff
    def reward(self, agent, neighbor_velocities, neighbor_positions):
        alignment_reward = 0
        cohesion_penalty = 0
        separation_penalty = 0
        out_of_flock_penalty = 0
        diversity_bonus = 0
        cohesion_bonus = 0
        angular_momentum_reward = 0
        formation_penalty = 0
        collision_penalty = 0
        out_of_flock = False
        midpoint = (SimulationVariables["SafetyRadius"] + SimulationVariables["NeighborhoodRadius"]) / 2

        # Circular formation incentive
        centroid = np.mean([a.position for a in self.agents], axis=0)
        radius = np.std([np.linalg.norm(a.position - centroid) for a in self.agents])

        # 1. Reward uniform angular distribution
        #angles = [np.arctan2(pos[1]-centroid[1], pos[0]-centroid[0])
        #          for pos in agent_positions]
        #angle_diff = np.std(np.diff(np.sort(angles)))
        #angular_momentum_reward += 2.0 / (1e-8 + angle_diff)

        # 2. Penalize irregular radial distribution
        formation_penalty -= 2 * radius

        # Centroid-based cohesion penalty
        distance_to_centroid = np.linalg.norm(agent.position - centroid)
        if distance_to_centroid > SimulationVariables["NeighborhoodRadius"]:
            cohesion_penalty -= 3 * distance_to_centroid
        else:
            cohesion_penalty += distance_to_centroid

        #speed_std = np.std([np.linalg.norm(a.velocity) for a in self.agents])
        #diversity_bonus += 0.2 * np.tanh(speed_std)
        if len(neighbor_velocities) > 1:
            diversity_bonus += 1.5 * len(neighbor_velocities)

        if len(neighbor_positions) < 2:
            # Reward if velocity points towards the centroid
            direction_to_centroid = (centroid - agent.position)
            direction_to_centroid /= np.linalg.norm(direction_to_centroid)  # Unit vector

            agent_velocity = agent.velocity / np.linalg.norm(agent.velocity)  # Unit vector

            alignment_with_centroid = np.dot(agent_velocity, direction_to_centroid)
            out_of_flock_penalty += 3 * alignment_with_centroid  # Scale reward based on alignment [-3, 3]
        else:
            # Alignment reward
            avg_velocity = np.mean(neighbor_velocities, axis=0)
            dot_product = np.dot(avg_velocity, agent.velocity)
            norm_product = np.linalg.norm(avg_velocity) * np.linalg.norm(agent.velocity)
            cos_angle = dot_product / norm_product if norm_product != 0 else 1.0
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            alignment_reward += 2 * (1 - 2*np.arccos(cos_angle) / np.pi)  # Range [-2, 2]


        if len(neighbor_positions) == 0:
            out_of_flock_penalty = -20  # Penalty for no neighbors
            #out_of_flock = True        #No for the time being
        else:
            # Separation penalty for close but non-colliding agents
            for i, pos in enumerate(neighbor_positions):
                distance = np.linalg.norm(agent.position - pos)
                if 0.5 * SimulationVariables["SafetyRadius"] <= distance < SimulationVariables["SafetyRadius"]:
                    # Linearly increasing penalty as distance decreases
                    separation_penalty -= 80 * (SimulationVariables["SafetyRadius"] - distance) / (0.5 * SimulationVariables["SafetyRadius"])
                elif midpoint - 0.1 *(SimulationVariables["NeighborhoodRadius"] - SimulationVariables["SafetyRadius"]) <= distance <= midpoint + 0.1 * (SimulationVariables["NeighborhoodRadius"] - SimulationVariables["SafetyRadius"]):
                    cohesion_bonus += 40
                elif SimulationVariables["SafetyRadius"] < distance <= midpoint:
                    cohesion_bonus += (5 / (midpoint - SimulationVariables["SafetyRadius"])) * (distance - SimulationVariables["SafetyRadius"])
                elif midpoint < distance <= SimulationVariables["NeighborhoodRadius"]:
                    cohesion_bonus += 15 - (10 / (SimulationVariables["NeighborhoodRadius"] - midpoint)) * (distance - midpoint)

                direction_to_neighbor = (pos - agent.position) / distance
                relative_velocity = neighbor_velocities[i] - agent.velocity
                closing_speed = np.dot(relative_velocity, direction_to_neighbor)
                if closing_speed > 0:  # Moving toward each other
                    time_to_collision = distance / closing_speed
                    collision_penalty = -2 / (1 + time_to_collision)

        total_reward = alignment_reward + cohesion_penalty + separation_penalty + out_of_flock_penalty + diversity_bonus + cohesion_bonus + formation_penalty + collision_penalty
        return total_reward, alignment_reward, cohesion_penalty, diversity_bonus, cohesion_bonus, separation_penalty, out_of_flock_penalty, formation_penalty, collision_penalty, out_of_flock

    def read_agent_locations(self):
        """Generate random initial positions for 6 agents"""
        return [
            [round(random.uniform(-2.5, 2.5), 1), round(random.uniform(-2.5, 2.5), 1)]  # [x, y] format
            for _ in range(6)
        ]

    def seed(self, seed=SimulationVariables["Seed"]):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    def save_reward_components(self, filename="reward_components.npy"):
      # Convert to numpy array for easy plotting
      reward_data = np.array([
          self.reward_components['alignment'],
          self.reward_components['cohesion_penalty'],
          self.reward_components['separation_penalty'],
          self.reward_components['diversity_bonus'],
          self.reward_components['cohesion_bonus'],
          self.reward_components['out_of_flock_penalty'],
          self.reward_components['formation_penalty'],
          self.reward_components['collsion_penalty'],
          self.reward_components['total'],
      ])

      np.save(filename, reward_data)
      print(f"Saved reward components to {filename}")

    def render(self, mode="human", save=False):
        # Create figure without GUI backend if not showing to humans
        if mode == "rgb_array":
            plt.switch_backend('agg')

        fig = plt.figure("Flocking Simulation", figsize=(10, 10))
        plt.clf()
        ax = fig.add_subplot(111)

        all_x = []
        all_y = []

        # Plot each agent as a point and velocity vectors
        positions = []  # Store all agent positions
        for agent in self.agents:
            pos = agent.position
            vel = agent.velocity
            all_x.append(pos[0])
            all_y.append(pos[1])
            positions.append(pos)

            # Plot agent position
            ax.scatter(pos[0], pos[1], color="blue",
                      label="Agent" if agent == self.agents[0] else "")

            # Plot velocity vector
            ax.arrow(pos[0], pos[1], vel[0]*0.05, vel[1]*0.05,
                    head_width=0.01, head_length=0.01,
                    fc="red", ec="red")

        # Draw connections between agents in distance range
        min_dist = 0.25
        max_dist = 2.0
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                pos1 = positions[i]
                pos2 = positions[j]
                distance = np.linalg.norm(pos1 - pos2)

                if SimulationVariables["SafetyRadius"] <= distance <= SimulationVariables["NeighborhoodRadius"]:
                    # Draw line between agents
                    ax.plot([pos1[0], pos2[0]],
                            [pos1[1], pos2[1]],
                            color='black',
                            alpha=0.4,
                            linestyle='--',
                            linewidth=0.8)

        # Dynamic window calculation
        x_center = np.mean(all_x)
        y_center = np.mean(all_y)
        margin = 1.2  # 20% margin
        window_size = 5 * margin

        ax.set_xlim(x_center - window_size, x_center + window_size)
        ax.set_ylim(y_center - window_size, y_center + window_size)

        ax.set_title(f"Flocking Simulation - Timestep {self.current_timestep}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

        # Handle different render modes
        if mode == "rgb_array":
                # Draw the figure first
                fig.canvas.draw()

                # Get the RGB buffer directly
                img = np.array(fig.canvas.renderer.buffer_rgba())

                # Convert RGBA to RGB
                img = img[..., :3]

                # Get correct dimensions
                w, h = fig.canvas.get_width_height()
                img = img.reshape((h, w, 3))

                plt.close(fig)
                return img
        elif mode == "human":
            if save:
                plt.savefig(f"frames/frame_{self.current_timestep:04d}.png")
            plt.pause(0.001)
        else:
            plt.close(fig)
            raise NotImplementedError(f"Render mode {mode} not supported")
new_env = FlockingEnv()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class MADDPG:
    def __init__(self, num_agents, obs_dim, action_dim, gamma=0.99, tau=0.01):
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.actors = [Actor(obs_dim, action_dim) for _ in range(num_agents)]
        self.critics = [Critic(obs_dim * num_agents + action_dim * num_agents) for _ in range(num_agents)]
        self.target_actors = [Actor(obs_dim, action_dim) for _ in range(num_agents)]
        self.target_critics = [Critic(obs_dim * num_agents + action_dim * num_agents) for _ in range(num_agents)]

        # 🔄 Add exploration parameters
        self.noise_processes = [
            OUNoise(action_dim,
                    theta=0.15 + 0.03*i,  # Vary per agent
                    sigma=0.25*(1 + 0.1*i))
            for i in range(num_agents)
        ]
        self.epsilon = 1.0  # For epsilon-greedy
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.agent_rewards = deque(maxlen=100)  # Track recent performance

        # Move networks to device
        self.actors = [actor.to(device) for actor in self.actors]
        self.critics = [critic.to(device) for critic in self.critics]
        self.target_actors = [actor.to(device) for actor in self.target_actors]
        self.target_critics = [critic.to(device) for critic in self.target_critics]

        for target_actor, actor in zip(self.target_actors, self.actors):
            target_actor.load_state_dict(actor.state_dict())
        for target_critic, critic in zip(self.target_critics, self.critics):
            target_critic.load_state_dict(critic.state_dict())

        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=3e-5) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=1e-4) for critic in self.critics]

        self.memory = ReplayBuffer(int(1e6))

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        # Convert to tensors and move to device
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32).to(device)
        action_batch = torch.tensor(np.array(action_batch), dtype=torch.float32).to(device)
        reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32).to(device)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32).to(device)
        done_batch = torch.tensor(np.array(done_batch), dtype=torch.float32).to(device)

        for agent_id in range(self.num_agents):
            critic = self.critics[agent_id]
            target_critic = self.target_critics[agent_id]

            with torch.no_grad():
                next_actions = []
                for i in range(self.num_agents):
                    agent_obs = next_state_batch[:, i*24 : (i+1)*24]
                    next_action = self.target_actors[i](agent_obs)
                    next_actions.append(next_action)
                next_actions = torch.cat(next_actions, dim=1)
                next_q_input = torch.cat([next_state_batch, next_actions], dim=1)
                target_q = target_critic(next_q_input).squeeze()
                target_q = reward_batch[:, agent_id] + self.gamma * (1 - done_batch) * target_q

            current_q_input = torch.cat([state_batch, action_batch], dim=1)
            current_q = critic(current_q_input).squeeze()
            critic_loss = nn.MSELoss()(current_q, target_q)

            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), 0.5)
            self.critic_optimizers[agent_id].step()

            actor = self.actors[agent_id]
            pred_actions = []
            for i in range(self.num_agents):
                agent_obs = state_batch[:, i*24 : (i+1)*24]
                if i == agent_id:
                    pred_action = self.actors[i](agent_obs)
                else:
                    pred_action = self.actors[i](agent_obs).detach()
                pred_actions.append(pred_action)
            pred_actions = torch.cat(pred_actions, dim=1)
            actor_loss = -critic(torch.cat([state_batch, pred_actions], dim=1)).mean()

            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), 0.3)
            self.actor_optimizers[agent_id].step()

            self.soft_update(self.actors[agent_id], self.target_actors[agent_id])
            self.soft_update(self.critics[agent_id], self.target_critics[agent_id])

        return actor_loss, critic_loss

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


    def get_action(self, obs, agent_id, training=True):
        """🔄 Unified action selection with exploration"""
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action = self.actors[agent_id](obs_tensor).cpu().numpy().flatten()

        if training:
            # 🔄 Hybrid exploration
            if np.random.rand() < self.epsilon:
                # Random exploration
                action = np.random.uniform(-1, 1, size=action.shape)
            else:
                # OU noise exploration
                noise_scale = 1.2 if len(self.agent_rewards) > 10 and np.mean(self.agent_rewards) < 0 else 0.8
                action += self.noise_processes[agent_id].sample(scale=noise_scale)

        return np.clip(action, -1, 1)

    def update_exploration(self, mean_agent_rewards):
        """🔄 Update based on per-agent performance"""
        self.agent_rewards.append(mean_agent_rewards)  # Now receives per-agent means
        avg_performance = np.mean(mean_agent_rewards)

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Adaptive noise scaling
        for i in range(self.num_agents):
            if mean_agent_rewards[i] < avg_performance:
                # Increase exploration for underperforming agents
                self.noise_processes[i].sigma = min(
                    self.noise_processes[i].sigma * 1.1,
                    0.5  # Max sigma
                )
            else:
                # Decrease exploration for good performers
                self.noise_processes[i].sigma *= 0.95

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2,
                 decay_rate=0.995, min_sigma=0.05):
        self.action_dim = action_dim
        self.mu = mu * np.ones(self.action_dim)
        self.theta = theta
        self.base_sigma = sigma
        self.sigma = sigma
        self.decay_rate = decay_rate
        self.min_sigma = min_sigma
        self.state = np.copy(self.mu)

    def reset(self):
        self.state = np.copy(self.mu)
        self.sigma = max(self.base_sigma * 0.5, self.min_sigma)  # 🔄 Partial reset

    def sample(self, scale=1.0):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        self.sigma = max(self.sigma * self.decay_rate, self.min_sigma)
        return self.state * scale  # 🔄 Add scaling parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = FlockingEnv()
num_agents = env.num_agents
obs_dim = env.observation_space[0].shape[0]  # Observation dimension per agent
action_dim = env.action_space[0].shape[0]    # Action dimension per agent

maddpg = MADDPG(num_agents, obs_dim, action_dim, 0.95, 0.02)

# Enhanced evaluation function
def evaluate_policy(maddpg, env, num, num_episodes=10, max_timesteps=300):
    total_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = {"__all__": False}
        frame_count = 0

        # Create video writer
        video_writer = imageio.get_writer(f'evaluation_{num}_{episode}.mp4', fps=30)

        while not done["__all__"] and frame_count < max_timesteps:
            # Render and save frame
            frame = env.render(mode='rgb_array')
            video_writer.append_data(frame)

            # Get actions from policy
            with torch.no_grad():
              actions = {i: maddpg.actors[i](torch.tensor(obs[i], dtype=torch.float32,device = device).unsqueeze(0)).cpu().detach().numpy().flatten()
                        for i in range(num_agents)}

            # Environment step
            obs, rewards, done, _ = env.step(actions)
            episode_reward += sum(rewards.values())
            frame_count += 1

        video_writer.close()
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode+1}: Reward {episode_reward:.1f}")

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward:.2f} ± {std_reward:.2f}")
    return avg_reward

num_episodes = 2500
max_timesteps = 250
batch_size = 256
a_loss, c_loss = [], []

# Reset noise for each episode
noise = {i: OUNoise(action_dim=2) for i in range(num_agents)}

for episode in range(num_episodes):
    obs = env.reset()
    agent_rewards = {i: [] for i in range(num_agents)}  # 🔄 Store individual rewards
    cumulative_reward = 0

    for t in range(max_timesteps):
        actions = {}
        for i in range(num_agents):
            # 🔄 Use unified action selection
            actions[i] = maddpg.get_action(obs[i], i, training=True)

        obs_next, rewards, done, _ = env.step(actions)

        full_obs = np.concatenate([obs[i] for i in range(num_agents)])
        full_actions = np.concatenate([actions[i] for i in range(num_agents)])
        full_rewards = [rewards[i] for i in range(num_agents)]
        full_next_obs = np.concatenate([obs_next[i] for i in range(num_agents)])
        done_flag = done.get("__all__", False)

        maddpg.memory.add((full_obs, full_actions, full_rewards, full_next_obs, done_flag))

        obs = obs_next
        for i in range(num_agents):
            agent_rewards[i].append(rewards[i])
        cumulative_reward += sum(rewards.values())
        loss = maddpg.update(batch_size)

        actor_loss, critic_loss = 0,0
        if loss:
            actor_loss, critic_loss = loss
            a_loss.append(actor_loss.cpu().detach().numpy())
            c_loss.append(critic_loss.cpu().detach().numpy())

        if done.get("__all__", False) or cumulative_reward < -30000:
            break

    print(f"Episode {episode + 1}/{num_episodes}, Reward: {cumulative_reward}, Actor: {actor_loss}, Critic: {critic_loss}")

    # 🔄 Calculate mean per-agent rewards
    mean_agent_rewards = [np.mean(agent_rewards[i]) for i in range(num_agents)]
    overall_mean = np.mean(mean_agent_rewards)

    # 🔄 Update exploration parameters
    maddpg.update_exploration(mean_agent_rewards)

    if (episode+1) % 50 == 0:
      evaluate_policy(maddpg, env, episode, 1)

    # 🔄 Modified model saving
    if (episode+1) % 100 == 0:
        for i in range(num_agents):
            torch.save({
                'actor': maddpg.actors[i].state_dict(),
                'critic': maddpg.critics[i].state_dict(),
                'noise': maddpg.noise_processes[i].__dict__,
                'epsilon': maddpg.epsilon
            }, f"agent_{i}_checkpoint_{episode+1}.pth")

    # 🔄 Performance-based noise reset
    for i in range(num_agents):
        if mean_agent_rewards[i] < overall_mean:
            maddpg.noise_processes[i].reset()

env.save_reward_components()
env.close()

import matplotlib.pyplot as plt

# Load reward data
reward_data = np.load("reward_components.npy")
components = [
    'Alignment', 'Cohesion Penalty', 'Separation Penalty', 'Diversity Bonus',
    'Cohesion Bonus', 'Out-of-Flock Penalty', 'Formation Penalty',
    'Collision Penalty', 'Total', 'Actor Loss', 'Critic Loss'
]

# Create subplots
fig, axs = plt.subplots(6, 2, figsize=(15, 15))  # 4 rows, 2 columns
fig.suptitle("Reward Component Breakdown", fontsize=16)

# Plot individual components
for i, (ax, label) in enumerate(zip(axs.flatten(), components)):
    if i == 9:
        ax.plot(a_loss, label=label, color='red')
    elif i == 10:
        ax.plot(c_loss, label=label, color='red')
    else:
        ax.plot(reward_data[i], label=label, color='blue')
    ax.set_title(label)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward Value")
    ax.grid(True)
    ax.legend()


# Remove the last empty subplot (if odd number of components)
if len(components) % 2 != 0:
    fig.delaxes(axs.flatten()[-1])

# Adjust layout
plt.tight_layout()
plt.savefig("reward_components_subplots.png")
plt.show()

import shutil
import os

# Define the directory to empty
frames_dir = "frames"  # Replace with your actual directory name

# Check if the directory exists
if os.path.exists(frames_dir):
  # Delete all files and subdirectories within the directory
  for filename in os.listdir(frames_dir):
      file_path = os.path.join(frames_dir, filename)
      try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)
          elif os.path.isdir(file_path):
              shutil.rmtree(file_path)
      except Exception as e:
          print('Failed to delete %s. Reason: %s' % (file_path, e))
  print(f"The directory '{frames_dir}' has been emptied.")
else:
  print(f"The directory '{frames_dir}' does not exist.")
  os.makedirs('frames')

num_agents = 6
num_episodes = 200

for i in range(num_agents):
    checkpoint = torch.load(f"agent_{i}_checkpoint_{num_episodes}.pth", weights_only=False)
    maddpg.actors[i].load_state_dict(checkpoint['actor'])


obs = new_env.reset()
new_env.current_timestep = 0
for t in range(300):
    new_env.render(save = True)

    actions = {i: maddpg.actors[i](torch.tensor(obs[i], dtype=torch.float32,device = device).unsqueeze(0)).cpu().detach().numpy().flatten()
               for i in range(num_agents)}
    obs, _, done, _ = new_env.step(actions)

    #if done["__all__"]:
        #break

new_env.close()

# prompt: take the and loop throiugh the frames in the frames dir and save them into a video

import os
import imageio

def save_video(output_path="flocking_simulation.mp4"):
    frames = []
    for filename in sorted(os.listdir('frames')):
        if filename.endswith('.png'):
            frame_path = os.path.join('frames', filename)
            frames.append(imageio.imread(frame_path))
    imageio.mimsave(output_path, frames, fps=30)

save_video()

#avg_reward = evaluate_policy(maddpg, new_env, 23, num_episodes=10)

