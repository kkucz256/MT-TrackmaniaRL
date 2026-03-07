"""
TMRL Agent wrapper dla SAC z stable-baselines3
Dziedziczy po TMRL ActorModule i TrainingAgent
"""
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from torch.utils.tensorboard import SummaryWriter
from tmrl.actor import TorchActorModule
from tmrl.training import TrainingAgent
import numpy as np


class TmrlSacActorModule(TorchActorModule):
    
    def __init__(self, observation_shape, action_shape, device='cuda', model_path=None, buffer_size=10000):
        try:
            super().__init__(observation_shape, action_shape, device)
            
            self.observation_shape = observation_shape
            self.action_shape = action_shape
            self.device_str = device
            self.buffer_size = buffer_size
            
            from gymnasium import spaces
            
            if isinstance(observation_shape, tuple):
                self.obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=observation_shape, dtype=np.float32)
            else:
                self.obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_shape,), dtype=np.float32)
            
            if isinstance(action_shape, int):
                self.action_space = spaces.Box(low=-1, high=1, shape=(action_shape,), dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)
            
            self.sac_model = None
            
            if model_path:
                self._load_sac_model(model_path, device)
            else:
                self._create_dummy_env_and_sac(device)
        
        except Exception as e:
            print(f"[Actor] Error: {e}")
            raise
    
    def _create_dummy_env_and_sac(self, device):
        try:
            from gymnasium import Env
            
            class DummyEnv(Env):
                def __init__(inner_self, obs_sp, act_sp):
                    inner_self.observation_space = obs_sp
                    inner_self.action_space = act_sp
                
                def reset(inner_self, seed=None, **kwargs):
                    return np.zeros(inner_self.observation_space.shape, dtype=np.float32), {}
                
                def step(inner_self, action):
                    return np.zeros(inner_self.observation_space.shape, dtype=np.float32), 0, False, False, {}
            
            dummy_env = DummyEnv(self.obs_space, self.action_space)
            
            self.sac_model = SAC(
                "MlpPolicy",
                dummy_env,
                learning_rate=3e-4,
                gamma=0.99,
                tau=0.005,
                buffer_size=self.buffer_size,
                batch_size=64,
                device=device,
                verbose=0
            )
            
            self.sac_model.set_logger(configure(folder=None, format_strings=[]))
            
        except Exception as e:
            print(f"[Actor] SAC creation failed: {e}")
            raise
    
    def _load_sac_model(self, path, device):
        try:
            self.sac_model = SAC.load(path, env=None, device=device)
            self.sac_model.set_logger(configure(folder=None, format_strings=[]))
        except Exception as e:
            print(f"[Actor] Load failed: {e}")
            raise
    
    def act(self, observation, test=True):
        obs_np = observation.cpu().numpy() if torch.is_tensor(observation) else observation
        action, _ = self.sac_model.predict(obs_np, deterministic=test)
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device_str)
        return action_tensor
    
    def save(self, path: str):
        self.sac_model.save(path)
    
    def load(self, path: str):
        self.sac_model = SAC.load(path)


class TmrlSacTrainingAgent(TrainingAgent):
    def __init__(self, observation_shape, action_shape, device='cuda', model_path=None, buffer_size=10000):
        try:
            self.train_steps = 0
            self.tb_writer = SummaryWriter(log_dir="./logs/trainer")
            
            self.actor_module = TmrlSacActorModule(observation_shape, action_shape, device, model_path, buffer_size)
            
            super().__init__(self.actor_module, self.actor_module.action_space, device)
        
        except Exception as e:
            print(f"[Trainer] Error: {e}")
            raise
    
    def get_actor(self):
        return self.actor_module
    
    def train(self, batch):
        try:
            if not batch or len(batch) == 0:
                return
            
            for exp in batch:
                try:
                    obs, action, reward, next_obs, done = exp
                    self.actor_module.sac_model.replay_buffer.add(
                        obs=obs,
                        action=action,
                        reward=float(reward),
                        next_obs=next_obs,
                        done=done,
                        infos=[{}]
                    )
                except Exception as e:
                    pass
            
            if self.actor_module.sac_model.replay_buffer.pos > 0:
                gradient_steps = max(1, min(len(batch) // 2, 10))
                try:
                    self.actor_module.sac_model.train(gradient_steps=gradient_steps)
                except Exception as e:
                    pass
            
            if self.train_steps % 100 == 0:
                try:
                    buf_size = self.actor_module.sac_model.replay_buffer.pos
                    self.tb_writer.add_scalar('Training/Buffer_Size', buf_size, self.train_steps)
                    self.tb_writer.add_scalar('Training/Learning_Rate', 3e-4, self.train_steps)
                    self.tb_writer.add_scalar('Training/Batch_Size', 64, self.train_steps)
                except Exception as e:
                    pass
            
            self.train_steps += 1
        
        except Exception as e:
            print(f"[Trainer] Error: {e}")
    
    def save(self, path: str):
        self.actor_module.save(path)
        self.tb_writer.flush()
    
    def load(self, path: str):
        self.actor_module.load(path)
