import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import multiprocessing as mp
import time
import numpy as np
import torch
import torch.nn as nn
import random
import os
import sys
from collections import deque

from trackmania_env import TrackmaniaEnv
from tmrl_sac_agent import TmrlSacTrainingAgent, TmrlSacActorModule
from gymnasium import spaces
from gymnasium.core import Wrapper
from torch.utils.tensorboard import SummaryWriter

GAMMA = 0.99
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
TRAIN_BATCH_THRESHOLD = 16
MEMORY_SIZE = 10000 
TAU = 0.005
AUTO_ENTROPY = True
WARMUP_STEPS = 2000

RESUME_TRAINING = False
MODEL_NAME = "sac_model_new"
MODEL_CHECKPOINT_DIR = "./models"

os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)

def collector_worker(experience_queue, run_flag):
    print("[COLLECTOR] Starting...")
    
    tb_writer = SummaryWriter(log_dir="./logs/collector")
    env = None
    
    try:
        time.sleep(3)
        env = TrackmaniaEnv()
        
        model_path = os.path.join(MODEL_CHECKPOINT_DIR, MODEL_NAME)
        
        if RESUME_TRAINING and os.path.exists(f"{model_path}.zip"):
            print(f"[COLLECTOR] Loading model: {model_path}")
            actor = TmrlSacActorModule(env.observation_space, env.action_space, device="cuda", model_path=model_path, buffer_size=MEMORY_SIZE)
        else:
            print(f"[COLLECTOR] Creating new actor")
            actor = TmrlSacActorModule(env.observation_space, env.action_space, device="cuda", buffer_size=MEMORY_SIZE)
        
        print("[COLLECTOR] Ready, collecting experiences")
        
        obs, _ = env.reset()
        steps = 0
        episode_rewards = []
        last_log = time.time()
        
        while run_flag.value == 1:
            if steps < WARMUP_STEPS:
                # Warmup phase: random actions
                action = env.action_space.sample()
            else:
                # Policy phase: model actions
                action_tensor = actor.act(obs, test=False)
                action = action_tensor.cpu().numpy().flatten()
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            experience = (obs, action, reward, next_obs, done)
            
            try:
                experience_queue.put(experience, timeout=0.5)
                episode_rewards.append(reward)
                steps += 1
                
                if time.time() - last_log >= 5.0:
                    avg_rew = np.mean(episode_rewards[-50:]) if episode_rewards else 0
                    print(f"[COLLECTOR] Steps: {steps:6d} | Reward: {avg_rew:.4f}")
                    tb_writer.add_scalar('Collector/Avg_Reward_50', avg_rew, steps)
                    tb_writer.add_scalar('Collector/Total_Steps', steps, steps)
                    tb_writer.flush()
                    last_log = time.time()
                    
            except mp.queues.Full:
                pass
            
            obs = next_obs
            if done:
                obs, _ = env.reset()
        
    except Exception as e:
        print(f"[COLLECTOR] ERROR: {e}")
    finally:
        tb_writer.flush()
        tb_writer.close()
        if env:
            env.close()
        print("[COLLECTOR] Closed.")

def learner_worker(experience_queue, run_flag):
    print("[TRAINER] Starting...")
    
    tb_writer = SummaryWriter(log_dir="./logs/learner")
    
    try:
        observation_space = spaces.Dict({
            "vision": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            "telemetry": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        })
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        model_path = os.path.join(MODEL_CHECKPOINT_DIR, MODEL_NAME)
        
        if RESUME_TRAINING and os.path.exists(f"{model_path}.zip"):
            print(f"[TRAINER] Loading model: {model_path}")
            training_agent = TmrlSacTrainingAgent(observation_space, action_space, device="cuda", model_path=model_path, buffer_size=MEMORY_SIZE)
        else:
            print(f"[TRAINER] Creating new training agent")
            training_agent = TmrlSacTrainingAgent(observation_space, action_space, device="cuda", buffer_size=MEMORY_SIZE)
        
        print("[TRAINER] Ready")
        
        memory = deque(maxlen=MEMORY_SIZE)
        batches_done = 0
        last_log_time = time.time()
        
        while run_flag.value == 1:
            try:
                experiences_batch = []
                try:
                    for _ in range(min(10, BATCH_SIZE)):
                        exp = experience_queue.get(timeout=0.1)
                        experiences_batch.append(exp)
                        memory.append(exp)
                except mp.queues.Empty:
                    pass
                
                if len(memory) >= TRAIN_BATCH_THRESHOLD:
                    try:
                        training_agent.train(experiences_batch if experiences_batch else list(memory)[-BATCH_SIZE:])
                        batches_done += 1
                        
                        if time.time() - last_log_time >= 2.0:
                            avg_reward = np.mean([exp[2] for exp in list(memory)[-min(100, len(memory)):]]) if memory else 0
                            print(f"[TRAINER] Batch: {batches_done:5d} | Reward: {avg_reward:.4f} | Buffer: {len(memory)}")
                            tb_writer.add_scalar('Learner/Avg_Reward_100', avg_reward, batches_done)
                            tb_writer.add_scalar('Learner/Buffer_Size', len(memory), batches_done)
                            tb_writer.add_scalar('Learner/Batches_Done', batches_done, batches_done)
                            tb_writer.flush()
                            last_log_time = time.time()
                        
                        if batches_done % 10 == 0:
                            training_agent.save(model_path)
                            print(f"[TRAINER] CHECKPOINT batch {batches_done}")
                    
                    except Exception as e:
                        print(f"[TRAINER] Train error: {e}")
                
            except Exception as e:
                print(f"[TRAINER] ERROR: {e}")
                break
        
        training_agent.save(model_path)
        print(f"[TRAINER] Final checkpoint saved")

    except Exception as e:
        print(f"[TRAINER] FATAL INITIALIZATION: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tb_writer.flush()
        tb_writer.close()
        print("[TRAINER] Closed.")



if __name__ == "__main__":
    print("\n" + "="*80)
    print("SAC LEARNER - TRACKMANIA")
    print("="*80)
    print(f"Learning Rate: {LEARNING_RATE} | Batch: {BATCH_SIZE} | Memory: {MEMORY_SIZE}")
    print(f"Gamma: {GAMMA} | Tau: {TAU} | Auto Entropy: {AUTO_ENTROPY}")
    print(f"Resume: {RESUME_TRAINING} | Model: {MODEL_NAME}")
    print("="*80 + "\n")
    
    mp.set_start_method('spawn', force=True)
    
    exp_queue = mp.Queue(maxsize=2000)
    run_flag = mp.Value('i', 1)
    
    p_learner = mp.Process(target=learner_worker, args=(exp_queue, run_flag), daemon=False)
    p_collector = mp.Process(target=collector_worker, args=(exp_queue, run_flag), daemon=False)
    
    try:
        p_learner.start()
        time.sleep(1)
        p_collector.start()
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        run_flag.value = 0
        
        for proc in [p_collector, p_learner]:
            try:
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.terminate()
            except:
                pass
        
        print("Done.")
        sys.exit(0)
