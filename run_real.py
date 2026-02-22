import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from tmrl.config.config_objects import ENV_CLS
from collections import deque
import gymnasium as gym

from cnn import TrackmaniaCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TM2020DiscreteWrapper(gym.Wrapper):
    """
    Wrapper, który tłumaczy świat rzeczywisty gry na warunki twojego uproszczonego DQN.
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(4)
        
    def step(self, action):
        mapping = {
            0: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            1: np.array([1.0, 0.0, -1.0], dtype=np.float32),
            2: np.array([1.0, 0.0, 1.0], dtype=np.float32),
            3: np.array([0.0, 1.0, 0.0], dtype=np.float32)
        }
        cont_action = mapping.get(int(action), np.array([0.0, 0.0, 0.0], dtype=np.float32))
        
        obs, reward, terminated, truncated, info = self.env.step(cont_action)
        
        img_obs = self._extract_image(obs)
        return img_obs, reward, terminated, truncated, info
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        img_obs = self._extract_image(obs)
        return img_obs, info

    def _extract_image(self, obs):
        if isinstance(obs, tuple):
            for item in reversed(obs):
                if isinstance(item, np.ndarray) and len(item.shape) >= 3:
                    return item
        elif isinstance(obs, np.ndarray) and len(obs.shape) >= 3:
            return obs
        return np.zeros((4, 64, 64), dtype=np.float32)

BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 2000
TARGET_UPDATE = 500
MEMORY_SIZE = 10000
LR = 1e-4

def train():
    base_env = ENV_CLS()
    env = TM2020DiscreteWrapper(base_env)
    
    n_actions = env.action_space.n

    policy_net = TrackmaniaCNN(input_channels=4, num_actions=n_actions).to(device)
    target_net = TrackmaniaCNN(input_channels=4, num_actions=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=MEMORY_SIZE)
    criterion = nn.SmoothL1Loss()

    steps_done = 0

    print("Rozpoczęcie treningu")
    for episode in range(100):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        done = False

        while not done:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1

            if random.random() > eps_threshold:
                with torch.no_grad():
                    action = policy_net(state).max(1)[1].view(1, 1)
            else:
                action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

            next_state_np, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            reward_t = torch.tensor([reward], device=device, dtype=torch.float32)
            next_state = torch.tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0) if not done else None

            memory.append((state, action, next_state, reward_t, torch.tensor([done], device=device, dtype=torch.bool)))
            state = next_state
            total_reward += reward

            if len(memory) >= BATCH_SIZE:
                transitions = random.sample(memory, BATCH_SIZE)
                
                batch_state = torch.cat([t[0] for t in transitions])
                batch_action = torch.cat([t[1] for t in transitions])
                batch_reward = torch.cat([t[3] for t in transitions])
                batch_done = torch.cat([t[4] for t in transitions])
                
                non_final_mask = ~batch_done
                non_final_next_states = torch.cat([t[2] for t in transitions if t[2] is not None])

                state_action_values = policy_net(batch_state).gather(1, batch_action)
                
                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                with torch.no_grad():
                    if non_final_mask.sum() > 0:
                        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
                
                expected_state_action_values = (next_state_values * GAMMA) + batch_reward

                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                optimizer.step()
                
                if steps_done % 50 == 0:
                    print(f"Krok: {steps_done} | Loss: {loss.item():.4f} | Akcja: {action.item()} | Epsilon: {eps_threshold:.2f}")

            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print(f"=== Epoka {episode} zakończona | Total Reward: {total_reward:.2f} ===")

if __name__ == "__main__":
    train()