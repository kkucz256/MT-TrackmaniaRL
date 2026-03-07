#!/usr/bin/env python3
"""
SAC LEARNER DLA TRACKMANII - DECOUPLED ARCHITECTURE
3 PROCESY: COLLECTOR + TRAINER (SAC) + Dashboard
Używa: stable-baselines3.SAC + TMRL pipeline + CNN
"""
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
import cv2
import datetime
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

RESUME_TRAINING = False
MODEL_NAME = "sac_model"
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
            actor = TmrlSacActorModule(env.observation_space, (9,), device="cuda", model_path=model_path, buffer_size=MEMORY_SIZE)
        else:
            print(f"[COLLECTOR] Creating new actor")
            actor = TmrlSacActorModule(env.observation_space, (9,), device="cuda", buffer_size=MEMORY_SIZE)
        
        print("[COLLECTOR] Ready, collecting experiences")
        
        obs, _ = env.reset()
        steps = 0
        episode_rewards = []
        last_log = time.time()
        
        while run_flag.value == 1:
            # Inference - act() obsługuje dict observation bezpośrednio
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
        env = TrackmaniaEnv()
        
        model_path = os.path.join(MODEL_CHECKPOINT_DIR, MODEL_NAME)
        
        if RESUME_TRAINING and os.path.exists(f"{model_path}.zip"):
            print(f"[TRAINER] Loading model: {model_path}")
            training_agent = TmrlSacTrainingAgent(env.observation_space, (9,), device="cuda", model_path=model_path, buffer_size=MEMORY_SIZE)
        else:
            print(f"[TRAINER] Creating new training agent")
            training_agent = TmrlSacTrainingAgent(env.observation_space, (9,), device="cuda", buffer_size=MEMORY_SIZE)
        
        print("[TRAINER] Ready")
        print("[TRAINER] RELOAD PLUGIN")
        env.close()
        
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

def dashboard_worker(experience_queue_peek, run_flag):
    print("[DASHBOARD] Starting reward monitor...")
    
    recent_rewards = deque(maxlen=50)
    
    while run_flag.value == 1:
        try:
            if not experience_queue_peek.empty():
                try:
                    exp = experience_queue_peek.get_nowait()
                    reward = exp[2]
                    recent_rewards.append(reward)
                    
                    display_frame = np.zeros((300, 500, 3), dtype=np.uint8)
                    
                    color = (0, 255, 0) if reward > 0 else (0, 0, 255)
                    cv2.putText(display_frame, f"Reward: {reward:.4f}", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                    
                    avg_rew = np.mean(recent_rewards) if recent_rewards else 0
                    cv2.putText(display_frame, f"Avg(50): {avg_rew:.4f}", (50, 180), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                    
                    cv2.putText(display_frame, f"Samples: {len(recent_rewards)}", (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                    
                    cv2.imshow("SAC Training - Rewards", display_frame)
                except:
                    pass
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            continue
    
    cv2.destroyAllWindows()
    print("[DASHBOARD] Closed")

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
    exp_queue_peek = mp.Queue(maxsize=100)
    run_flag = mp.Value('i', 1)
    
    p_learner = mp.Process(target=learner_worker, args=(exp_queue, run_flag), daemon=False)
    p_collector = mp.Process(target=collector_worker, args=(exp_queue, run_flag), daemon=False)
    p_dashboard = mp.Process(target=dashboard_worker, args=(exp_queue_peek, run_flag), daemon=True)
    
    try:
        p_learner.start()
        time.sleep(1)
        p_collector.start()
        time.sleep(1)
        p_dashboard.start()
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        run_flag.value = 0
        
        for proc in [p_collector, p_learner, p_dashboard]:
            try:
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.terminate()
            except:
                pass
        
        print("Done.")
        sys.exit(0)
