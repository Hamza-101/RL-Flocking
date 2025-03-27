import os
import gym
import json
import numpy as np
import torch as th
from tqdm import tqdm
from Settings import *
from gym import spaces
import matplotlib
matplotlib.use('Agg')  # faster than rendering plots
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, get_system_info
from stable_baselines3.common.callbacks import BaseCallback
from scipy.signal import savgol_filter
import shutil
from PlotAnimationRL import *
import glob
from scipy.spatial import cKDTree
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
import itertools
from datetime import datetime
import sys

positions_directory = "Results/Flocking/Testing/Episodes"  

policy_kwargs = dict(
    activation_fn=th.nn.Tanh,  
     # directly pass dict instead of list containing dict
    net_arch=dict(pi=[256, 256], vf=[256, 256])  
)

class Encoder(json.JSONEncoder):
    def default(self, obj):
        return json.JSONEncoder.default(self, obj)  

class TQDMProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(TQDMProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_print = 0
        self.print_interval = total_timesteps // 100 

    def _on_training_start(self) -> None:
        print(f"Starting training: 0/{self.total_timesteps} (0%)")
        
    def _on_step(self) -> bool:
        if (self.model.num_timesteps - self.last_print) >= self.print_interval:
            pct = int(100 * self.model.num_timesteps / self.total_timesteps)
            print(f"\nTraining Progress: {pct}% ({self.model.num_timesteps}/{self.total_timesteps})")
            self.last_print = self.model.num_timesteps
        return True
  
class Agent:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.acceleration = np.zeros(2) 
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"]
        self.velocity = np.round(np.random.uniform(
            -SimulationVariables["VelocityUpperLimit"]/5, 
            SimulationVariables["VelocityUpperLimit"]/5, 
            size=2), 2)       
        self.max_velocity = SimulationVariables["VelocityUpperLimit"]


    def update(self, action):
        previous_acceleration = np.copy(self.acceleration)
        self.acceleration = action  
        # replace instead of add
        acc_magnitude = np.linalg.norm(self.acceleration)   
        # hard limit
        if acc_magnitude > SimulationVariables["AccelerationUpperLimit"]:
            self.acceleration = (self.acceleration / acc_magnitude) * SimulationVariables["AccelerationUpperLimit"]
        # acceleration smoothing
        alpha = 0.7
        self.acceleration = alpha * previous_acceleration + (1-alpha) * self.acceleration
        self.velocity += self.acceleration * SimulationVariables["dt"]
        vel = np.linalg.norm(self.velocity)
        if vel > 0:
            if vel > self.max_velocity:
                self.velocity = (self.velocity / vel) * self.max_velocity 

        self.position += self.velocity * SimulationVariables["dt"]
        
        return self.position, self.velocity

class FlockingEnv(gym.Env):
    def __init__(self, seed=None):
        super(FlockingEnv, self).__init__()
        self.seed(seed)
        self.episode=0
        self.counter=0
        self.CTDE=False
        self.current_timestep = 0
        self.reward_log = []
        self.np_random, _ = gym.utils.seeding.np_random(None)
        self.cumulative_rewards = {i: 0 for i in range(SimulationVariables["SimAgents"])}
        self.isolation_counter = 0
        self.num_configs = self._count_available_configs()
        
        # cdte reward, write to disk at end
        self.alignment_rewards_buffer = []
        self.cohesion_rewards_buffer = []
        
        self.agents = [Agent(position) for position in self.read_agent_locations()]

        min_action = np.array([-5, -5] * len(self.agents), dtype=np.float32)
        max_action = np.array([5, 5] * len(self.agents), dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

        min_obs = np.array([-np.inf, -np.inf, -2.5, -2.5] * len(self.agents), dtype=np.float32)
        max_obs = np.array([np.inf, np.inf, 2.5, 2.5] * len(self.agents), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]
        
    def step(self, actions):
        training_rewards = {}
        noise = NormalActionNoise(mean=np.zeros(len(actions)), sigma=0.1 * np.ones(len(actions)))
        noisy_actions = actions + noise()
        self.current_timestep += 1
        reward=0
        done=False
        info={}
        observations = self.simulate_agents(noisy_actions)
        reward, out_of_flock = self.calculate_reward()
        reward = reward/10 

        if self.CTDE == False and out_of_flock:
            self.isolation_counter += 1
            # 30 timesteps to allow rejoin flock, if not then end ep
            if self.isolation_counter > 30:  
                done = True
        else:
            # counter reset when all together
            self.isolation_counter = 0

        return observations, reward, done, info

    def reset(self):   
        self.seed(SimulationVariables["Seed"])
        self.agents = [Agent(position) for position in self.read_agent_locations()]
        for agent in self.agents:
            agent.acceleration = np.zeros(2)
            # initial veloctiies a little bit smaller
            agent.velocity = np.round(np.random.uniform(
                -SimulationVariables["VelocityUpperLimit"]/5, 
                SimulationVariables["VelocityUpperLimit"]/5, 
                size=2), 2)
            
                         

        observation = self.get_observation().flatten()
        self.current_timestep = 0  
        return observation   

    def close(self):
        print("Simulation is complete. Cleaned Up!.")
        
    def simulate_agents(self, actions):
        observations = []  
        actions_reshaped = actions.reshape(((SimulationVariables["SimAgents"]), 2))
        for i, agent in enumerate(self.agents):
            position, velocity = agent.update(actions_reshaped[i])
            observation_pair = np.concatenate([position, velocity])
            observations = np.concatenate([observations, observation_pair])
            
        return observations
    
    def check_collision(self, agent):
        for other in self.agents:
            if agent != other:
                distance = np.linalg.norm(agent.position - other.position)
                if distance < SimulationVariables["SafetyRadius"]:
                    return True  
        return False

    def get_observation(self):
        n_agents = len(self.agents)
        #print(f"Getting observation for {n_agents} agents")
        obs = np.zeros((n_agents, 4), dtype=np.float32)

        for i, agent in enumerate(self.agents):
            obs[i] = [
                agent.position[0],
                agent.position[1],
                agent.velocity[0],
                agent.velocity[1]
            ]
            #print(f"Agent {i} pos: {agent.position}, vel: {agent.velocity}")

        return obs
   
    def get_closest_neighbors(self, agent):
        # neighborhood radius
        radius = SimulationVariables["NeighborhoodRadius"]        
        # get agents
        neighbors = [a for a in self.agents if a != agent]
        neighbor_positions = []
        neighbor_velocities = []
        
        for neighbor in neighbors:
            dist = np.linalg.norm(agent.position - neighbor.position)
            if dist <= radius:
                neighbor_positions.append(neighbor.position)
                neighbor_velocities.append(neighbor.velocity)
        
        return neighbor_positions, neighbor_velocities    
   
    def calculate_reward(self):
        total_reward = 0
        out_of_flock = False
        cumulative_alignment = 0
        cumulative_cohesion = 0

        for i, agent in enumerate(self.agents):
            neighbor_positions, neighbor_velocities = self.get_closest_neighbors(agent)
            agent_reward, alignment_reward, cohesion_reward, out_of_flock = self.reward(agent, neighbor_velocities, neighbor_positions)
            self.cumulative_rewards[i] += agent_reward
            cumulative_alignment += alignment_reward
            cumulative_cohesion += cohesion_reward

            total_reward += agent_reward
            
        if(self.CTDE==True):
            # memory buffer
            self.alignment_rewards_buffer.append(cumulative_alignment)
            self.cohesion_rewards_buffer.append(cumulative_cohesion)

        return total_reward, out_of_flock

    # flush buffer to disk at end of ep
    def save_reward_buffers(self):
        if self.CTDE and self.alignment_rewards_buffer and self.cohesion_rewards_buffer:
            os.makedirs(positions_directory, exist_ok=True)
            
            with open(os.path.join(positions_directory, f"CohesionRewardsEpisode{self.episode}.json"), "w") as f:
                for reward in self.cohesion_rewards_buffer:
                    f.write(f"{reward}\n")
                    
            with open(os.path.join(positions_directory, f"AlignmentRewardsEpisode{self.episode}.json"), "w") as f:
                for reward in self.alignment_rewards_buffer:
                    f.write(f"{reward}\n")
                    
            # clear buffer
            self.alignment_rewards_buffer = []
            self.cohesion_rewards_buffer = []

    def reward(self, agent, neighbor_velocities, neighbor_positions):
        CohesionReward = 0
        AlignmentReward = 0
        SeparationReward = 0
        total_reward = 0
        outofflock = False

        c_weight = SimulationVariables["RewardWeights"]["cohesion"]
        a_weight = SimulationVariables["RewardWeights"]["alignment"]
        s_weight = SimulationVariables["RewardWeights"]["separation"]
        
        # increase cohesion importance towards end, try and maintain cohesion
        time_factor = min(1.0, self.current_timestep / 150)  # at 150 timsteps
        cohesion_time_boost = 1.0 + 0.5 * time_factor  # [1, 1.5] # not used anymore because just gives free reward
        alignment_time_decay = 1.0 - 0.3 * time_factor  # [1, 0.7]


        if len(neighbor_positions) > 0:
            # scale factor for number of neighbors, optimal 3 neigbors (wont affect centre)
            neighbor_factor = len(neighbor_positions) / (SimulationVariables["SimAgents"] - 1)
            neighbor_factor = 3 if neighbor_factor > 3 else neighbor_factor
    
            # get all pairwise distances and 3 closest
            distances = [np.linalg.norm(agent.position - np.array(pos)) for pos in neighbor_positions]
            closest_distances = sorted(distances)[:min(3, len(distances))]
            optimal_distance = SimulationVariables["SafetyRadius"] * 1.5
        
            # closest neighbor reward to give optimal spacing
            neighbor_spacing_rewards = []
            for dist in closest_distances:
                # bell curve, 1 at optimal distance, falls of both sides
                bell_curve_precision = 5.0 + 10.0 * time_factor  # From 5.0 to 15.0
                spacing_reward = np.exp(-bell_curve_precision * (dist - optimal_distance)**2 / optimal_distance**2)
                neighbor_spacing_rewards.append(spacing_reward)
        
            # whole group cohesion
            group_center = np.mean(neighbor_positions, axis=0)
            d_to_center = np.linalg.norm(agent.position - group_center)
            cohesion_strictness = 0.5 + 0.5 * time_factor  # From 0.5 to 1.0
            global_cohesion = np.exp(-cohesion_strictness * d_to_center)
            
            # combine spacing and cohesion -> prioritise global group
            CohesionReward = 10.0 * (0.6 * np.mean(neighbor_spacing_rewards) + 0.4 * global_cohesion) * neighbor_factor
            
            avg_velocity = np.mean(neighbor_velocities, axis=0)
            agent_vel_norm = agent.velocity / (np.linalg.norm(agent.velocity) + 1e-6)
            avg_vel_norm = avg_velocity / (np.linalg.norm(avg_velocity) + 1e-6)
            direction_alignment = np.dot(agent_vel_norm, avg_vel_norm)  # [-1, 1]
            
            # speed matching
            agent_speed = np.linalg.norm(agent.velocity)
            avg_speed = np.linalg.norm(avg_velocity)
            speed_diff = np.abs(agent_speed - avg_speed)
            speed_alignment = 1.0 - min(speed_diff / SimulationVariables["VelocityUpperLimit"], 1.0)  # [0, 1]
            # combine alignment, speed matching and number of neighbors -> not used anymore ?
            AlignmentReward = 8.0 * (direction_alignment + 0.3 * speed_alignment) * neighbor_factor #* alignment_time_decay ** neighbor_factor 
            
            # separation -> test with 0 
            for neighbor_pos in neighbor_positions:
                distance = np.linalg.norm(agent.position - neighbor_pos)
                if distance < SimulationVariables["SafetyRadius"]:
                    SeparationReward -= 0.5 * (1.0 - distance / SimulationVariables["SafetyRadius"])
                   # SeparationReward=0

            # reset isolation
            agent.isolation_time = 0
            
        else:
            outofflock = True
            
            # check duration of isolation
            if not hasattr(agent, 'isolation_time'):
                for a in self.agents:
                    a.isolation_time = 0
                    
            agent.isolation_time += 1
            
            # penalty incr with time
            isolation_penalty = -5.0 - 2.0 * agent.isolation_time
            CohesionReward = isolation_penalty
            
            # recovering from isolation mechanism
            # find all pairwise distances, move towards centre of 3 agents closest to each other
            # reward for moving in that direction
            agent_positions = np.array([a.position for a in self.agents])
            distance_matrix = np.zeros((len(self.agents), len(self.agents)))

            for i in range(len(self.agents)):
                for j in range(i+1, len(self.agents)):
                    distance_matrix[i,j] = distance_matrix[j,i] = np.linalg.norm(agent_positions[i] - agent_positions[j])
            
            min_distance = float('inf')
            flock_core_indices = []
            
            for trio in itertools.combinations(range(len(self.agents)), 3):
                trio_distance = (distance_matrix[trio[0], trio[1]] + 
                                distance_matrix[trio[0], trio[2]] + 
                                distance_matrix[trio[1], trio[2]])
                if trio_distance < min_distance:
                    min_distance = trio_distance
                    flock_core_indices = list(trio)
            
            flock_center = np.mean([self.agents[i].position for i in flock_core_indices], axis=0)
            direction_to_flock = flock_center - agent.position
            distance_to_flock = np.linalg.norm(direction_to_flock)
            CohesionReward -= 3.0 * np.tanh(distance_to_flock)
            
            # recovery reward
            if distance_to_flock > 0:
                direction_to_flock = direction_to_flock / distance_to_flock
                agent_direction = agent.velocity / (np.linalg.norm(agent.velocity) + 1e-6)
                moving_toward_flock = np.dot(agent_direction, direction_to_flock)
                AlignmentReward = 4.0 * moving_toward_flock - 5.0
        
        # scale to normalise values
        CohesionReward = np.tanh(CohesionReward/5.0)
        AlignmentReward = np.tanh(AlignmentReward/5.0)
        SeparationReward = np.tanh(SeparationReward/5.0)

                # Instead of boosting cohesion reward, rebalance the weights
        effective_c_weight = c_weight * (1.0 + time_factor)  # Increase over time
        effective_a_weight = a_weight * (1.0 - time_factor)  # Decrease over time

        # Then use these effective weights instead of time-based multipliers inside the reward
        total_reward = effective_c_weight * CohesionReward + effective_a_weight * AlignmentReward + s_weight * SeparationReward

        return total_reward, AlignmentReward, CohesionReward, outofflock

    def _count_available_configs(self):
        """Count how many configuration directories are available"""
        base_dir = os.path.dirname(Results["InitPositions"])
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
            self._generate_default_configs()
            return 10  
        
        count = 0
        while os.path.exists(f"{base_dir}/Config_{count}"):
            count += 1
        
        if count == 0:
            self._generate_default_configs()
            return 10
        
        return count
    
    def _generate_default_configs(self):
        """Generate default configuration files if none exist"""
        print("No configuration files found. Generating default configurations...")
        try:
            from generate_configs import generate_agents_config
            generate_agents_config(
                num_agents=SimulationVariables["SimAgents"],
                num_configs=10,
                radius=0.8,
                random_seed=SimulationVariables["Seed"]
            )
            print("Default configurations generated successfully.")
        except ImportError:
            print("ERROR: Could not import generate_configs module.")
            print("Please run 'python generate_configs.py' manually.")
            raise

    def read_agent_locations(self):
        config_idx = self.counter % self.num_configs
        File = os.path.join(Results["InitPositions"] + str(config_idx), "config.json")
        
        try:
            with open(File, "r") as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"WARNING: Config file {File} not found. Generating new configs...")
            self._generate_default_configs()
            with open(File, "r") as f:
                data = json.load(f)
            return data

def delete_files(): 
    Paths = [
        "Results/Flocking/Testing/Dynamics/Accelerations",
        "Results/Flocking/Testing/Dynamics/Velocities",
        "Results/Flocking/Testing/Rewards/Other"
    ]

    Logs = [
        "AlignmentReward_log.json", "CohesionReward_log.json",
        "SeparationReward_log.json", "CollisionReward_log.json",
        "Reward_Total_log.json"
    ]

    for Path in Paths:
        for episode in range(0, 10):
            file_path = os.path.join(Files['Flocking'], Path, f"Episode{episode}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")

    for log_file in Logs:
        for episode in range(0, 10):
            file_path = os.path.join(Files['Flocking'], "Testing", "Rewards", "Components", f"Episode{episode}", log_file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")

    print("All specified files have been deleted.")

def setup_episode_folder(episode_name):
    episode_folder = os.path.join(positions_directory, episode_name)
    if os.path.exists(episode_folder):
        for file in os.listdir(episode_folder):
            os.remove(os.path.join(episode_folder, file)) 
    else:
        os.makedirs(episode_folder, exist_ok=True)
    return episode_folder

positions_directory = "Results/Flocking/Testing/Episodes" 

def generate_combined():
    directory = "Results/Flocking/Testing/Episodes"
    os.makedirs(directory, exist_ok=True)  

    fig_combined, ax_combined = plt.subplots(figsize=(12, 8))
    ax_combined_seconds = ax_combined.twiny()  

    cohesion_files = sorted([f for f in os.listdir(directory) if f.startswith("CohesionRewardsEpisode")])
    alignment_files = sorted([f for f in os.listdir(directory) if f.startswith("AlignmentRewardsEpisode")])

    episodes = sorted(set(
        f.split("CohesionRewardsEpisode")[1].split(".json")[0] for f in cohesion_files
    ).intersection(
        f.split("AlignmentRewardsEpisode")[1].split(".json")[0] for f in alignment_files
    ))

    if not episodes:
        print("No valid episodes found!")
        return

    for episode in episodes:
        cohesion_file = os.path.join(directory, f"CohesionRewardsEpisode{episode}.json")
        alignment_file = os.path.join(directory, f"AlignmentRewardsEpisode{episode}.json")

        if not os.path.exists(cohesion_file) or not os.path.exists(alignment_file):
            print(f"Skipping missing files for Episode {episode}")
            continue

        with open(cohesion_file, "r") as f:
            cohesion_rewards = [float(line.strip()) for line in f.readlines()][:200]
        with open(alignment_file, "r") as f:
            alignment_rewards = [float(line.strip()) for line in f.readlines()][:200]

        combined_rewards = [c + a for c, a in zip(cohesion_rewards, alignment_rewards)]

        timesteps = range(1, len(combined_rewards) + 1)
        seconds = [timestep / 10 for timestep in timesteps]  

        fig, ax = plt.subplots(figsize=(10, 6))
        ax_seconds = ax.twiny()  

        ax.plot(timesteps, cohesion_rewards, label="Cohesion Rewards", alpha=0.7, linestyle='--')
        ax.plot(timesteps, alignment_rewards, label="Alignment Rewards", alpha=0.7, linestyle='-.')
        ax.plot(timesteps, combined_rewards, label="Combined Rewards", alpha=0.7)

        ax.set_title(f"Rewards for Episode {episode}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True)

        ax.set_xlim(1, 200)
        ax_seconds.set_xlim(ax.get_xlim())
        ax_seconds.set_xticks(ax.get_xticks())
        ax_seconds.set_xticklabels([f"{tick / 10:.1f}" for tick in ax.get_xticks()])
        ax_seconds.set_xlabel("Time (seconds)")

        plt.tight_layout()
        plt.savefig(os.path.join(positions_directory, f"Episode_{episode}_Rewards.png"), dpi=300)
        plt.close(fig)

        ax_combined.plot(timesteps, combined_rewards, label=f"Combined Rewards (Episode {episode})", alpha=0.5)

    ax_combined.set_title("Combined Rewards - All Episodes (200 Timesteps)")
    ax_combined.set_xlabel("Timestep")
    ax_combined.set_ylabel("Reward")
    ax_combined.legend()
    ax_combined.grid(True)

    ax_combined.set_xlim(1, 200)
    ax_combined_seconds.set_xlim(ax_combined.get_xlim())
    ax_combined_seconds.set_xticks(ax_combined.get_xticks())
    ax_combined_seconds.set_xticklabels([f"{tick / 10:.1f}" for tick in ax_combined.get_xticks()])
    ax_combined_seconds.set_xlabel("Time (seconds)")

    plt.tight_layout()
    combined_plot_path = os.path.join(positions_directory, "Combined_Cohesion_Alignment_Rewards.png")
    plt.savefig(combined_plot_path, dpi=300)
    plt.close(fig_combined)
    
def generateVelocity(episode, episode_folder):
    velocities_dict = {}
    velocity_file_path = os.path.join(positions_directory, f"Episode{episode}_velocities.json")
    
    if not os.path.exists(velocity_file_path):
        print(f"File {velocity_file_path} not found.")
        return

    with open(velocity_file_path, 'r') as f:
        episode_velocities = json.load(f)

    for agent_id in range(6):  
        velocities_dict.setdefault(agent_id, []).extend(episode_velocities.get(str(agent_id), []))

    colors = plt.cm.get_cmap('tab10', 6)
    downsample_factor = 10  

    plt.figure(figsize=(12, 6))
    plt.clf()
    
    for agent_id in range(6):  
        agent_velocities = np.array(velocities_dict[agent_id])
        agent_velocities = savgol_filter(agent_velocities, window_length=3, polyorder=2, axis=0)  
        velocities_magnitude = np.sqrt(agent_velocities[:, 0]**2 + agent_velocities[:, 1]**2)
        
        plt.plot(velocities_magnitude[::downsample_factor], label=f"Agent {agent_id+1}", color=colors(agent_id), linewidth=1)

    plt.title(f"Velocity - Episode {episode}")
    plt.xlabel("Time Step")
    plt.ylabel("Velocity Magnitude")
    plt.ylim([0, 5])  
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(episode_folder, f"Episode_{episode}_Velocity.png"))
    plt.close()  
    print(f"Velocity plot saved for Episode {episode}")

def generateAcceleration(episode, episode_folder):
    acceleration_file_path = os.path.join(positions_directory, f"Episode{episode}_accelerations.json")
    
    if not os.path.exists(acceleration_file_path):
        print(f"File {acceleration_file_path} not found.")
        return

    with open(acceleration_file_path, 'r') as f:
        episode_accelerations = json.load(f)

    colors = plt.cm.get_cmap('tab10', 6)
    downsample_factor = 10  

    plt.figure(figsize=(12, 6))
    plt.clf()

    for agent_id in range(6):
        agent_accelerations = np.array(episode_accelerations[str(agent_id)])
        smoothed_accelerations = np.sqrt(agent_accelerations[:, 0]**2 + agent_accelerations[:, 1]**2)
        smoothed_accelerations = savgol_filter(smoothed_accelerations, window_length=15, polyorder=3, axis=0)  

        plt.plot(smoothed_accelerations[::downsample_factor], label=f"Agent {agent_id+1}", color=colors(agent_id), linewidth=1)

    plt.title(f"Acceleration - Episode {episode}")
    plt.xlabel("Time Step")
    plt.ylabel("Acceleration Magnitude")
    plt.ylim([0, 10])  
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(episode_folder, f"Episode_{episode}_Acceleration.png"))
    plt.close()  
    print(f"Acceleration plot saved for Episode {episode}")

def generatePlots():
    for episode in range(SimulationVariables["Episodes"]):
        episode_name = f"Episode{episode}".split('_')[0]
        episode_folder = setup_episode_folder(episode_name)
        
        generateVelocity(episode, episode_folder)
        generateAcceleration(episode, episode_folder)

def delete_existing_files(directory, pattern):
    files = glob.glob(os.path.join(directory, pattern))
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
#------------------------
class LossCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LossCallback, self).__init__(verbose)
        self.loss_threshold = 2000

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) >= 1000:
            recent_losses = [ep_info['loss'] for ep_info in self.model.ep_info_buffer[-1000:]]
            average_loss = np.mean(recent_losses)

            if average_loss < self.loss_threshold:
                print(f"Stopping training because average loss ({average_loss}) is below threshold.")
                return False  

        return True

class AdaptiveExplorationCallback(BaseCallback):
    def __init__(self, initial_ent_coef=0.40, min_ent_coef=1e-3, decay_rate=0.85, max_reward_threshold=60, verbose=0):
        super(AdaptiveExplorationCallback, self).__init__(verbose)
        self.initial_ent_coef = initial_ent_coef       
        self.min_ent_coef = min_ent_coef               
        self.decay_rate = decay_rate                   
        self.ent_coef = initial_ent_coef               
        self.max_reward_threshold = max_reward_threshold  

    def _on_training_start(self):
        self.model.ent_coef = self.initial_ent_coef

    def _on_step(self) -> bool:
        # cumulative rewards
        all_cumulative_rewards = self.model.env.get_attr('cumulative_rewards')
        
        # check env for any cumulative rewards above threshold
        any_env_above = any(
            all(reward >= self.max_reward_threshold for reward in env_rewards.values())
            for env_rewards in all_cumulative_rewards
        )
        
        if any_env_above:
            self.ent_coef = max(self.ent_coef * self.decay_rate, self.min_ent_coef)
        else:
            self.ent_coef = self.initial_ent_coef
        self.model.ent_coef = self.ent_coef
        return True
#------------------------
def run_hyperparameter_search():
    """
    Run a grid search over cohesion and alignment reward weights.
    Tests all combinations and identifies the best performing ones.
    """
    print("\n==== Starting Hyperparameter Search ====")
    
    cohesion_values = SimulationVariables["HyperparamSearch"]["cohesion"]
    alignment_values = SimulationVariables["HyperparamSearch"]["alignment"]
    episodes_per_combo = SimulationVariables["HyperparamSearch"]["episodes_per_combo"]
    timesteps_per_run = SimulationVariables["HyperparamSearch"]["timesteps_per_run"]
    eval_episodes = SimulationVariables["HyperparamSearch"]["eval_episodes"]
    search_dir = f"{Files['Flocking']}/HyperparamSearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(search_dir, exist_ok=True)
    
    with open(f"{search_dir}/search_config.json", "w") as f:
        json.dump({
            "cohesion_values": cohesion_values,
            "alignment_values": alignment_values,
            "episodes_per_combo": episodes_per_combo,
            "timesteps_per_run": timesteps_per_run,
            "eval_episodes": eval_episodes
        }, f, indent=4)

    results = {}
    
    def make_env():
        env = FlockingEnv()
        env.CTDE = False  # disable CTDE during training
        return env

    param_combinations = list(itertools.product(cohesion_values, alignment_values))
    print(f"Testing {len(param_combinations)} hyperparameter combinations...")
    
    for i, (c_weight, a_weight) in enumerate(param_combinations):
        print(f"\n[{i+1}/{len(param_combinations)}] Testing cohesion={c_weight:.2f}, alignment={a_weight:.2f}")
        
        SimulationVariables["RewardWeights"]["cohesion"] = c_weight
        SimulationVariables["RewardWeights"]["alignment"] = a_weight
    
        if device == "cuda":
            n_envs = 6
        else:
            n_envs = 6
        
        vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_reward=5.0)
        
        model = PPO(
            policy, 
            vec_env, 
            device=device,
            policy_kwargs=policy_kwargs,
            verbose=0,  
            learning_rate=1e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.3,
            vf_coef=0.5,
            max_grad_norm=0.5
        )
        model.set_random_seed(SimulationVariables["ModelSeed"])
        
        progress_callback = TQDMProgressCallback(total_timesteps=timesteps_per_run)
        model.learn(total_timesteps=timesteps_per_run, callback=progress_callback)
        
        combo_dir = f"{search_dir}/cohesion_{c_weight:.2f}_alignment_{a_weight:.2f}"
        os.makedirs(combo_dir, exist_ok=True)
        
        model.save(f"{combo_dir}/model")
        
        eval_env = FlockingEnv()
        eval_rewards = []
        
        for episode in range(eval_episodes):
            obs = eval_env.reset()
            eval_env.CTDE = True
            done = False
            ep_reward = 0
            
            while not done and eval_env.current_timestep < SimulationVariables["EvalTimeSteps"]:
                actions, _ = model.predict(obs)
                obs, reward, done, _ = eval_env.step(actions)
                ep_reward += reward
                
            eval_rewards.append(ep_reward)
            print(f"  Evaluation episode {episode+1}/{eval_episodes} reward: {ep_reward:.2f}")

        avg_reward = sum(eval_rewards) / len(eval_rewards)
        std_reward = np.std(eval_rewards)
        results[(c_weight, a_weight)] = {
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "rewards": eval_rewards
        }
        
        with open(f"{combo_dir}/eval_results.json", "w") as f:
            json.dump({
                "cohesion_weight": c_weight,
                "alignment_weight": a_weight,
                "avg_reward": avg_reward,
                "std_reward": std_reward,
                "rewards": eval_rewards
            }, f, indent=4)
        
        print(f"  Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        
        vec_env.close()
        eval_env.close()
    
    best_params = max(results.items(), key=lambda x: x[1]["avg_reward"])
    best_c, best_a = best_params[0]
    best_reward = best_params[1]["avg_reward"]
    
    print("\n==== Hyperparameter Search Results ====")
    print(f"Best cohesion weight: {best_c:.2f}")
    print(f"Best alignment weight: {best_a:.2f}")
    print(f"Best average reward: {best_reward:.2f}")
    
    plt.figure(figsize=(12, 10))
    heatmap_data = np.zeros((len(cohesion_values), len(alignment_values)))
    
    for i, c in enumerate(cohesion_values):
        for j, a in enumerate(alignment_values):
            if (c, a) in results:
                heatmap_data[i, j] = results[(c, a)]["avg_reward"]
    
    plt.imshow(heatmap_data, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(label='Average Reward')
    plt.xlabel('Alignment Weight Index')
    plt.ylabel('Cohesion Weight Index')
    plt.title('Reward Heatmap: Cohesion vs Alignment Weights')
    
    plt.xticks(range(len(alignment_values)), [f"{a:.1f}" for a in alignment_values], rotation=90)
    plt.yticks(range(len(cohesion_values)), [f"{c:.1f}" for c in cohesion_values])
    
    best_i = cohesion_values.index(best_c)
    best_j = alignment_values.index(best_a)
    plt.scatter(best_j, best_i, color='red', marker='*', s=200, label=f'Best: C={best_c:.2f}, A={best_a:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{search_dir}/heatmap.png", dpi=300)
    plt.close()
    
    with open(f"{search_dir}/all_results.json", "w") as f:
        json_results = {f"{c:.2f}_{a:.2f}": {
            "avg_reward": float(data["avg_reward"]),
            "std_reward": float(data["std_reward"]),
            "rewards": [float(r) for r in data["rewards"]]
        } for (c, a), data in results.items()}
        
        json.dump({
            "results": json_results,
            "best_params": {
                "cohesion": float(best_c),
                "alignment": float(best_a),
                "reward": float(best_reward)
            }
        }, f, indent=4)
    
    print("\nUpdating SimulationVariables with best parameters...")
    SimulationVariables["RewardWeights"]["cohesion"] = best_c
    SimulationVariables["RewardWeights"]["alignment"] = best_a
    
    return best_c, best_a, best_reward

#------------------------
if __name__ == "__main__":
    env = FlockingEnv()
    
    if os.path.exists(Results["Rewards"]):
        os.remove(Results["Rewards"])
        print(f"File {Results['Rewards']} has been deleted.")

    if os.path.exists("training_rewards.json"):
        os.remove("training_rewards.json")
        print(f"File training_rewards has been deleted.")    

    def seed_everything(seed):
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        th.backends.cudnn.deterministic = True
        env.seed(seed)
        env.action_space.seed(seed)

    env = FlockingEnv()
    seed_everything(SimulationVariables["Seed"])

    loss_callback = LossCallback()
    adaptive_exploration_callback = AdaptiveExplorationCallback(initial_ent_coef=0.05,    # decreased ent coef
                                    min_ent_coef=1e-7, decay_rate=0.8, max_reward_threshold=20  
                                    )
    progress_callback = TQDMProgressCallback(total_timesteps=SimulationVariables["LearningTimeSteps"])

    device = "cuda" if th.cuda.is_available() else "cpu"
    device = 'cpu'
    print(f"Using device: {device}")

    if device == "cpu":
        th.set_num_threads(10)

    policy = "MlpPolicy"
    ModelName = f"{Files['Flocking']}/Models/Flocking_{policy}_{SimulationVariables['LearningTimeSteps']}"

    
    if device == "cuda":
        n_envs = 6
    else:
        n_envs = 6

    def make_env():
        env = FlockingEnv()
        env.CTDE = False  # disable cdte during training
        return env
        
    env = SubprocVecEnv([make_env for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    # Check if we should load an existing model
    if SimulationVariables["LoadModel"]:
        model_path = SimulationVariables["ModelPath"]
        print(f"Loading existing model from {model_path}")
        
        # Load the model
        model = PPO.load(
            model_path, 
            env=env,
            device=device,
            # Optionally override parameters when loading
            # tensorboard_log="./ppo_flocking_tensorboard/"
        )
        
        # Optional: Print model architecture
        print(f"Loaded model architecture: {model.policy}")
        
        # If continuing training, optionally adjust learning rate for fine-tuning
        if SimulationVariables["ContinueTraining"]:
            print("Will continue training loaded model")
            # Optionally reduce learning rate for fine-tuning
            model.learning_rate = 5e-5  # Lower learning rate for fine-tuning
    
    model = PPO(
            policy, 
            env, 
            device=device,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./ppo_flocking_tensorboard/",
            verbose=1,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=0.7, 
            max_grad_norm= 1.0  # prevent exploding gradients
        )
    

    model.set_random_seed(SimulationVariables["ModelSeed"])

    # Only train if not loading model or if continuing training
    if not SimulationVariables["LoadModel"] or SimulationVariables["ContinueTraining"]:
        model.learn(
            total_timesteps=SimulationVariables["LearningTimeSteps"],  
            callback=[progress_callback, adaptive_exploration_callback]
        )
        # Save after training
        model.save(f"{Files['Flocking']}/Models/Flocking_{policy}_Simulation")
    
    env = FlockingEnv()

    model = PPO.load(f"{Files['Flocking']}/Models/Flocking_{policy}_Simulation", device=device)

    delete_files()
    os.makedirs(f"{Files['Flocking']}/Models", exist_ok=True)
    os.makedirs(positions_directory, exist_ok=True)

    env.counter=0
    episode_rewards_dict = {}
    positions_dict = {i: [] for i in range(len(env.agents))}

    delete_existing_files(positions_directory, "CohesionRewardsEpisode*.json")
    delete_existing_files(positions_directory, "AlignmentRewardsEpisode*.json")
 
    for episode in tqdm(range(0, SimulationVariables["Episodes"])):
        env.episode = episode
        obs = env.reset()
        env.CTDE = True  # enable ctde during eval
        done = False
        timestep = 0
        reward_episode = []

        distances_dict = []
        positions_dict = {i: [] for i in range(len(env.agents))}
        velocities_dict = {i: [] for i in range(len(env.agents))}
        accelerations_dict = {i: [] for i in range(len(env.agents))}
        trajectory_dict = {i: [] for i in range(len(env.agents))}
        
        print(f"\n--- Episode {episode} ---")  
        print(env.counter)

        for i, agent in enumerate(env.agents):
            accelerations_dict[i].append(agent.acceleration.tolist())
            velocities_dict[i].append(agent.velocity.tolist())
            positions_dict[i].append(agent.position.tolist())
            trajectory_dict[i].append(agent.position.tolist())

        while timestep < SimulationVariables["EvalTimeSteps"]:
            actions, state = model.predict(obs)
            obs, reward, done, info = env.step(actions)
            reward_episode.append(reward)
            
            timestep_distances = {}  
            
            for i, agent in enumerate(env.agents):
                positions_dict[i].append(agent.position.tolist())
                velocity = agent.velocity.tolist()
                velocities_dict[i].append(velocity)
                acceleration = agent.acceleration.tolist()
                accelerations_dict[i].append(acceleration)
                trajectory_dict[i].append(agent.position.tolist())
                
                distances = []
                for j, other_agent in enumerate(env.agents):
                    if i != j:  
                        distance = np.linalg.norm(np.array(other_agent.position) - np.array(agent.position))
                        distances.append(distance)
                timestep_distances[i] = distances

            distances_dict.append(timestep_distances)

            timestep += 1
            episode_rewards_dict[str(episode)] = reward_episode
            
        # write reward buffer at end of ep
        env.save_reward_buffers()

        with open(os.path.join(positions_directory, f"Episode{episode}_positions.json"), 'w') as f:
            json.dump(positions_dict, f, indent=4)
        with open(os.path.join(positions_directory, f"Episode{episode}_velocities.json"), 'w') as f:
            json.dump(velocities_dict, f, indent=4)
        with open(os.path.join(positions_directory, f"Episode{episode}_accelerations.json"), 'w') as f:
            json.dump(accelerations_dict, f, indent=4)
        with open(os.path.join(positions_directory, f"Episode{episode}_distances.json"), 'w') as f:
            json.dump(distances_dict, f, indent=4)  
        with open(os.path.join(positions_directory, f"Episode{episode}_trajectory.json"), 'w') as f:
            json.dump(trajectory_dict, f, indent=4)

        env.counter += 1
        print(sum(reward_episode))

    with open(rf"{Results['EpisodalRewards']}.json", 'w') as f:
        json.dump(episode_rewards_dict, f, indent=4)

    env.close()

    generatePlots()
    generate_combined()
    
    if SimulationVariables["RunHyperparamSearch"]:
        best_cohesion, best_alignment, _ = run_hyperparameter_search()
        print(f"Using best hyperparameters from search: cohesion={best_cohesion:.2f}, alignment={best_alignment:.2f}")
