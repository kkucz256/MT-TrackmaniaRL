import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tmrl.training_offline import TorchTrainingOffline
from tmrl.actor import ActorModule
from cnn import TrackmaniaCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNActorModule(ActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        self.net = TrackmaniaCNN(input_channels=4, num_actions=4)
        
    def forward(self, obs):
        return self.net(obs)
        
    def act(self, obs, test=False):
        img_obs = obs[3] if isinstance(obs, tuple) else obs
        obs_tensor = torch.tensor(img_obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.net(obs_tensor)
            action_idx = q_values.max(1)[1].item()
        if not test and random.random() < 0.1: 
            action_idx = random.randint(0, 3)
        return action_idx

class DQNTrainingAgent(TorchTrainingOffline):
    def __init__(self, observation_space, action_space, device):
        super().__init__(
            observation_space=observation_space, 
            action_space=action_space, 
            device=device,
            epochs=1000,
            rounds=10,
            steps=50,
            start_training=128 
        )

        self.policy_net = DQNActorModule(observation_space, action_space).to(self.device)
        self.target_net = DQNActorModule(observation_space, action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.criterion = nn.SmoothL1Loss()
        self.gamma = 0.99
        self.updates_done = 0
        self.target_update_freq = 500
        self.csv_file = "training_metrics.csv"
        self.best_model_path = "dqn_best_weights.pth"
        if not os.path.isfile(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Step", "Loss", "Avg_Rew"])

    def train(self, batch):
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = batch
        state = torch.tensor(obs_batch[3] if isinstance(obs_batch, tuple) else obs_batch, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_obs_batch[3] if isinstance(next_obs_batch, tuple) else next_obs_batch, dtype=torch.float32, device=self.device)
        action = torch.tensor(act_batch, dtype=torch.long, device=self.device).view(-1, 1)
        reward = torch.tensor(rew_batch, dtype=torch.float32, device=self.device)
        done = torch.tensor(done_batch, dtype=torch.bool, device=self.device)
        
        state_action_values = self.policy_net(state).gather(1, action)
        next_state_values = torch.zeros(state.size(0), device=self.device)
        non_final_mask = ~done
        if non_final_mask.sum() > 0:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(next_state[non_final_mask]).max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.updates_done += 1
        
        # Log metrics every update
        loss_val = loss.item()
        avg_rew = reward.mean().item()
        
        csv_path = "training_metrics.csv"
        
        # Check if file exists, if not create with header
        if not os.path.isfile(csv_path):
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Step", "Loss", "Avg_Reward"])
        
        # Append metrics to CSV
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.updates_done, loss_val, avg_rew])
        
        print(f"[TRAIN] Step: {self.updates_done} | Loss: {loss_val:.6f} | Avg_Reward: {avg_rew:.6f}")

        if self.updates_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"[UPDATE] Target network updated at step {self.updates_done}")
            
        return {"loss": loss_val}

    def get_actor(self):
        return self.policy_net