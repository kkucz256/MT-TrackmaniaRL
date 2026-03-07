import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.utils.tensorboard import SummaryWriter
from tmrl.actor import TorchActorModule
from tmrl.training import TrainingAgent
from gymnasium import spaces
import numpy as np


class MultimodalFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        self.vision_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        cnn_output_size = 64 * 4 * 4
        telemetry_input_size = observation_space["telemetry"].shape[0]
        
        self.telemetry_mlp = nn.Sequential(
            nn.Linear(telemetry_input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        
        telemetry_output_size = 32
        
        total_input = cnn_output_size + telemetry_output_size
        
        self.fusion = nn.Sequential(
            nn.Linear(total_input, features_dim),
            nn.ReLU(),
        )
        
        print(f"[MultimodalExtractor] Vision CNN output: {cnn_output_size}")
        print(f"[MultimodalExtractor] Telemetry MLP output: {telemetry_output_size}")
        print(f"[MultimodalExtractor] Fusion output: {features_dim}")
    
    def forward(self, observations):
        vision = observations["vision"]
        telemetry = observations["telemetry"]
        
        if vision.dtype != torch.float32:
            vision = vision.float()
        if vision.shape[-1] == 3:
            vision = vision.permute(0, 3, 1, 2)
        vision = vision / 255.0 if vision.max() > 1.0 else vision
        
        vision_features = self.vision_cnn(vision)
        vision_features = vision_features.reshape(vision_features.shape[0], -1)  # Flatten
        
        if telemetry.dtype != torch.float32:
            telemetry = telemetry.float()
        telemetry_features = self.telemetry_mlp(telemetry)
        
        combined = torch.cat([vision_features, telemetry_features], dim=1)
        output = self.fusion(combined)
        
        return output


class TmrlSacActorModule(TorchActorModule):
    
    def __init__(self, observation_space, action_shape, device='cuda', model_path=None, buffer_size=10000):
        try:
            super().__init__(observation_space, action_shape, device)
            
            self.observation_space_spec = observation_space
            self.action_shape = action_shape
            self.device_str = device
            self.buffer_size = buffer_size
            
            if isinstance(observation_space, spaces.Dict):
                self.obs_space = observation_space
            else:
                if isinstance(observation_space, tuple):
                    self.obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=observation_space, dtype=np.float32)
                else:
                    self.obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_space,), dtype=np.float32)
            
            # Handle action space - can be Box or tuple/int
            if isinstance(action_shape, spaces.Box):
                self.action_space = action_shape
            elif isinstance(action_shape, int):
                self.action_space = spaces.Box(low=-1, high=1, shape=(action_shape,), dtype=np.float32)
            elif isinstance(action_shape, tuple):
                self.action_space = spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # Default fallback
            
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
                    if isinstance(inner_self.observation_space, spaces.Dict):
                        return {k: np.zeros(v.shape, dtype=np.float32) for k, v in inner_self.observation_space.spaces.items()}, {}
                    else:
                        return np.zeros(inner_self.observation_space.shape, dtype=np.float32), {}
                
                def step(inner_self, action):
                    if isinstance(inner_self.observation_space, spaces.Dict):
                        obs = {k: np.zeros(v.shape, dtype=np.float32) for k, v in inner_self.observation_space.spaces.items()}
                    else:
                        obs = np.zeros(inner_self.observation_space.shape, dtype=np.float32)
                    return obs, 0, False, False, {}
            
            dummy_env = DummyEnv(self.obs_space, self.action_space)
            
            policy_kwargs = {
                'features_extractor_class': MultimodalFeaturesExtractor,
                'features_extractor_kwargs': {'features_dim': 256},
                'net_arch': [256, 256],
            }
            
            self.sac_model = SAC(
                "MultiInputPolicy" if isinstance(self.obs_space, spaces.Dict) else "MlpPolicy",
                dummy_env,
                learning_rate=3e-4,
                gamma=0.99,
                tau=0.005,
                buffer_size=self.buffer_size,
                batch_size=64,
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=0
            )
            
            self.sac_model.set_logger(configure(folder=None, format_strings=[]))
            
            params = sum(p.numel() for p in self.sac_model.policy.parameters())
            print(f"[Actor] Created SAC with {params:,} parameters")
            
        except Exception as e:
            print(f"[Actor] SAC creation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_sac_model(self, path, device):
        try:
            self.sac_model = SAC.load(path, env=None, device=device)
            self.sac_model.set_logger(configure(folder=None, format_strings=[]))
        except Exception as e:
            print(f"[Actor] Load failed: {e}")
            raise
    
    def act(self, observation, test=True):
        if isinstance(observation, dict):
            obs_dict = {}
            for k, v in observation.items():
                if torch.is_tensor(v):
                    obs_dict[k] = v.cpu().numpy() if v.device.type != 'cpu' else v.numpy()
                else:
                    obs_dict[k] = v
            action, _ = self.sac_model.predict(obs_dict, deterministic=test)
        else:
            obs_np = observation.cpu().numpy() if torch.is_tensor(observation) else observation
            action, _ = self.sac_model.predict(obs_np, deterministic=test)
        
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device_str)
        return action_tensor
    
    def save(self, path: str):
        self.sac_model.save(path)
    
    def load(self, path: str):
        self.sac_model = SAC.load(path)


class TmrlSacTrainingAgent(TrainingAgent):
    def __init__(self, observation_space, action_shape, device='cuda', model_path=None, buffer_size=10000):
        try:
            self.train_steps = 0
            self.tb_writer = SummaryWriter(log_dir="./logs/trainer")
            
            self.actor_module = TmrlSacActorModule(observation_space, action_shape, device, model_path, buffer_size)
            
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
