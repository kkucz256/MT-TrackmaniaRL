import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import multiprocessing as mp
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import random
import sys
from collections import deque

from trackmania_env import TrackmaniaEnv
from tmrl_sac_agent import TmrlSacTrainingAgent, TmrlSacActorModule
from gymnasium import spaces
from gymnasium.core import Wrapper
from torch.utils.tensorboard import SummaryWriter

GAMMA = 0.99
LEARNING_RATE = 3e-4  # Back to 3e-4 (2e-4 was too conservative)
BATCH_SIZE = 64
TRAIN_BATCH_THRESHOLD = 16
MEMORY_SIZE = 10000 
TAU = 0.005
AUTO_ENTROPY = True
ENT_COEF = 0.1  # Reduced back to ~default (0.2 was causing too much randomness)
WARMUP_STEPS = 1500  # Balanced: more than 500, less than 2000

RESUME_TRAINING = False
MODEL_NAME = "sac_model_new"
MODEL_CHECKPOINT_DIR = "./models"

os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)

def collector_worker(experience_queue, run_flag, logs_dir):
    print("[COLLECTOR] Starting...")
    
    collector_log_dir = os.path.join(logs_dir, "collector")
    os.makedirs(collector_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=collector_log_dir)
    env = None
    
    try:
        time.sleep(3)
        env = TrackmaniaEnv()
        
        model_path = os.path.join(MODEL_CHECKPOINT_DIR, MODEL_NAME)
        
        # Usuń stary model jeśli nie wznawiam treningu
        if not RESUME_TRAINING and os.path.exists(f"{model_path}.zip"):
            print(f"[COLLECTOR] Removing old model: {model_path}.zip")
            try:
                os.remove(f"{model_path}.zip")
            except Exception as e:
                print(f"[COLLECTOR] Failed to remove old model: {e}")
        
        if RESUME_TRAINING and os.path.exists(f"{model_path}.zip"):
            print(f"[COLLECTOR] Loading model: {model_path}")
            actor = TmrlSacActorModule(env.observation_space, env.action_space, device="cuda", model_path=model_path, buffer_size=MEMORY_SIZE)
        else:
            print(f"[COLLECTOR] Creating new actor")
            actor = TmrlSacActorModule(env.observation_space, env.action_space, device="cuda", buffer_size=MEMORY_SIZE)
        
        print("[COLLECTOR] Ready, collecting experiences")
        
        obs, _ = env.reset()
        steps = 0
        current_ep_reward = 0.0  
        current_ep_length = 0    # NOWE: Długość pojedynczego przejazdu
        episode_returns = []     
        episode_lengths = []     # NOWE: Lista długości zakończonych przejazdów
        last_log = time.time()
        last_sync = time.time()  # NOWE: Śledzenie ostatniej synchronizacji wag
        
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
            
            # NOWE: Synchronizacja wag co 100 kroków
            if steps % 100 == 0 and steps > 0 and os.path.exists(f"{model_path}.zip"):
                try:
                    actor.load(model_path)
                    sync_elapsed = time.time() - last_sync
                    print(f"[COLLECTOR] Synchronized weights at step {steps} (took {sync_elapsed:.3f}s)")
                    last_sync = time.time()
                except Exception as e:
                    print(f"[COLLECTOR] Failed to sync weights: {e}")
            
            current_ep_reward += reward  
            current_ep_length += 1       # NOWE: Inkrementacja długości
            
            experience = (obs, action, reward, next_obs, done)
            
            try:
                experience_queue.put(experience, timeout=0.5)
                steps += 1
                
                if done:
                    episode_returns.append(current_ep_reward)
                    episode_lengths.append(current_ep_length)  # NOWE
                    print(f"\n[EPISODE] #{len(episode_returns)} ZAKOŃCZONY | Return: {current_ep_reward:+.2f} | Steps: {current_ep_length} | Status: {'WIN' if current_ep_reward > 50 else 'FAIL'}\n")
                    current_ep_reward = 0.0  
                    current_ep_length = 0                      # NOWE
                
                if time.time() - last_log >= 5.0 and len(episode_returns) > 0:
                    avg_return = np.mean(episode_returns[-50:]) if len(episode_returns) >= 1 else 0
                    avg_length = np.mean(episode_lengths[-50:]) if len(episode_lengths) >= 1 else 0
                    max_return = np.max(episode_returns[-50:]) if len(episode_returns) >= 1 else 0
                    print(f"[COLLECTOR] Steps: {steps:6d} | Avg Ep Return: {avg_return:.2f} | Avg Len: {avg_length:.1f} | Max: {max_return:.2f}")
                    tb_writer.add_scalar('Collector/Episode_Return_Mean_50', avg_return, len(episode_returns))
                    tb_writer.add_scalar('Collector/Episode_Length_Mean_50', avg_length, len(episode_returns))
                    tb_writer.add_scalar('Collector/Episode_Return_Max_50', max_return, len(episode_returns))
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

def learner_worker(experience_queue, run_flag, logs_dir):
    print("[TRAINER] Starting...")
    
    learner_log_dir = os.path.join(logs_dir, "learner")
    os.makedirs(learner_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=learner_log_dir)
    
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
                            buf_size = training_agent.actor_module.sac_model.replay_buffer.size()
                            print(f"[TRAINER] Batch: {batches_done:5d} | Buffer: {buf_size}")
                            
                            tb_writer.add_scalar('Training/Replay_Buffer_Size', buf_size, batches_done)
                            tb_writer.add_scalar('Training/Batches_Done', batches_done, batches_done)
                            
                            sb3_logger = training_agent.actor_module.sac_model.logger
                            if sb3_logger is not None and hasattr(sb3_logger, 'name_to_value'):
                                metrics = sb3_logger.name_to_value
                                if 'train/actor_loss' in metrics:
                                    tb_writer.add_scalar('Training_Loss/Actor_Loss', metrics['train/actor_loss'], batches_done)
                                if 'train/critic_loss' in metrics:
                                    tb_writer.add_scalar('Training_Loss/Critic_Loss', metrics['train/critic_loss'], batches_done)
                                if 'train/ent_coef' in metrics:
                                    tb_writer.add_scalar('Training_Params/Entropy_Coef', metrics['train/ent_coef'], batches_done)
                                if 'train/learning_rate' in metrics:
                                    tb_writer.add_scalar('Training_Params/Learning_Rate', metrics['train/learning_rate'], batches_done)
                            
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
    
    # Utwórz wersjonowany katalog logów
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_base_dir = "./logs"
    os.makedirs(logs_base_dir, exist_ok=True)
    versioned_logs_dir = os.path.join(logs_base_dir, timestamp)
    os.makedirs(versioned_logs_dir, exist_ok=True)
    print(f"[MAIN] Logs saved to: {versioned_logs_dir}\n")
    
    mp.set_start_method('spawn', force=True)
    
    exp_queue = mp.Queue(maxsize=2000)
    run_flag = mp.Value('i', 1)
    
    p_learner = mp.Process(target=learner_worker, args=(exp_queue, run_flag, versioned_logs_dir), daemon=False)
    p_collector = mp.Process(target=collector_worker, args=(exp_queue, run_flag, versioned_logs_dir), daemon=False)
    
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
