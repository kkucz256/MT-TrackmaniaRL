import sys
import argparse
import json
import numpy as np
import gymnasium as gym
import torch
import os

from tmrl.actor import ActorModule
ActorModule.load = lambda self, path, device: self

class DummyEnv:
    pass
sys.modules['__main__'].DummyEnv = DummyEnv

import tmrl.config.config_objects as cfg_obj
import tmrl.config.config_constants as cfg
from agent import DQNTrainingAgent, DQNActorModule

cfg.MODEL_PATH = "" 
cfg.REDIS_HSET_NAME = "final_fix_v4"
cfg.CHECKPOINT_PATH = os.path.join(os.getcwd(), "fresh_checkpoint.pth")
cfg.DATASET_PATH = os.path.join(os.getcwd(), "fresh_dataset.pkl")

def safe_load_state_dict(self, state_dict, strict=True):
    return None

torch.nn.Module.load_state_dict = safe_load_state_dict

class TM2020DiscreteWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(4)
        self.idle_counter = 0
    def step(self, action):
        mapping = {0: [1.0,0.0,0.0], 1: [1.0,0.0,-1.0], 2: [1.0,0.0,1.0], 3: [0.0,1.0,0.0]}
        cont_action = np.array(mapping.get(int(action), [0.0,0.0,0.0]), dtype=np.float32)
        obs, rew, term, trunc, info = self.env.step(cont_action)
        self.idle_counter = self.idle_counter + 1 if rew < 0.1 else 0
        done = term or trunc or self.idle_counter >= 3
        r = rew - 0.05 + (-10.0 if self.idle_counter >= 3 else 0.0)
        return self._extract_image(obs), r, done, False, info
    def reset(self, **kwargs):
        self.idle_counter = 0
        obs, info = self.env.reset(**kwargs)
        return self._extract_image(obs), info
    def _extract_image(self, obs):
        img = obs[3] if isinstance(obs, tuple) else obs
        return img if (isinstance(img, np.ndarray) and len(img.shape) >= 3) else np.zeros((4,64,64), dtype=np.float32)

cfg_obj.ENV_CLS = lambda: TM2020DiscreteWrapper(cfg_obj.ENV_CLS())
cfg_obj.TRAINING_AGENT_CLS = DQNTrainingAgent
cfg_obj.ACTOR_MODULE_CLS = DQNActorModule

from tmrl.__main__ import main as tmrl_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--trainer', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--install', action='store_true')
    
    args, _ = parser.parse_known_args()
    
    expected_flags = ['install', 'test', 'benchmark', 'record_reward', 'check_env', 'wandb', 'expert', 'use_keyboard']
    for flag in expected_flags:
        if not hasattr(args, flag):
            setattr(args, flag, False)
    if not hasattr(args, 'config'):
        args.config = {}

    import tmrl.config.config_objects as co
    co.ACTOR_MODULE_CLS = DQNActorModule
    co.TRAINING_AGENT_CLS = DQNTrainingAgent
    
    tmrl_main(args)