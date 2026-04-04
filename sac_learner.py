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
from tmrl_sac_agent import TmrlSacTrainingAgent, TmrlSacActorModule, MODEL_SIZE_CONFIG
from gymnasium import spaces
from gymnasium.core import Wrapper
from torch.utils.tensorboard import SummaryWriter

#Model parameters
GAMMA = 0.99
LEARNING_RATE = 3e-4
MODEL_SIZE = "Large"  # Options: "Small", "Base", "Large"
BATCH_SIZE = 64
TRAIN_BATCH_THRESHOLD = 1500
MEMORY_SIZE = 50000 
TAU = 0.005
AUTO_ENTROPY = True
ENT_COEF = 0.1
WARMUP_STEPS = 1500

#Training settings
RESUME_TRAINING = False
PRETRAINED_MODEL_PATH = ""
MODEL_CHECKPOINT_DIR = "./models"
MAX_STEPS = 800000
MODEL_NAME = f"model_mix_track_01_06{MODEL_SIZE}_{MAX_STEPS}steps_buf{MEMORY_SIZE}"
CHECKPOINT_STEPS = 20000


os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)

def collector_worker(experience_queue, run_flag, logs_dir, learner_buffer_size):
    print("[COLLECTOR] Starting...")
    
    collector_log_dir = os.path.join(logs_dir, "collector")
    os.makedirs(collector_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=collector_log_dir)
    env = None
    
    try:
        time.sleep(3)
        env = TrackmaniaEnv()
        
        model_path = os.path.join(MODEL_CHECKPOINT_DIR, MODEL_NAME)
        
        if not RESUME_TRAINING and os.path.exists(f"{model_path}.zip"):
            print(f"[COLLECTOR] Removing old model: {model_path}.zip")
            try:
                os.remove(f"{model_path}.zip")
            except Exception as e:
                print(f"[COLLECTOR] Failed to remove old model: {e}")
        
        if RESUME_TRAINING:
            if os.path.exists(f"{model_path}.zip"):
                print(f"[COLLECTOR] Resuming from target model: {model_path}")
                actor = TmrlSacActorModule(
                    env.observation_space,
                    env.action_space,
                    device="cuda",
                    model_path=model_path,
                    buffer_size=MEMORY_SIZE,
                    model_size=MODEL_SIZE,
                )
            elif os.path.exists(f"{PRETRAINED_MODEL_PATH}.zip"):
                print(f"[COLLECTOR] Loading pretrained model: {PRETRAINED_MODEL_PATH}")
                actor = TmrlSacActorModule(
                    env.observation_space,
                    env.action_space,
                    device="cuda",
                    model_path=PRETRAINED_MODEL_PATH,
                    buffer_size=MEMORY_SIZE,
                    model_size=MODEL_SIZE,
                )
            else:
                print("[COLLECTOR] No target or pretrained model found. Creating new actor")
                actor = TmrlSacActorModule(
                    env.observation_space,
                    env.action_space,
                    device="cuda",
                    buffer_size=MEMORY_SIZE,
                    model_size=MODEL_SIZE,
                )
        else:
            print(f"[COLLECTOR] Creating new actor")
            actor = TmrlSacActorModule(
                env.observation_space,
                env.action_space,
                device="cuda",
                buffer_size=MEMORY_SIZE,
                model_size=MODEL_SIZE,
            )
        
        print("[COLLECTOR] Waiting for active telemetry connection...")
        while run_flag.value == 1:
            _, tele = env.pipeline.get_state()
            if tele:
                print(f"\n[COLLECTOR] Telemetry lock acquired! Received live signal from the game: {tele}")
                break
            
            print("[COLLECTOR] No telemetry yet. Press 'Reload' in OpenPlanet (F3). Waiting...")
            time.sleep(2.0)
            
        print("[COLLECTOR] Ready, collecting experiences")
        
        obs, _ = env.reset()
        steps = 0
        current_ep_reward = 0.0  
        current_ep_length = 0
        episode_returns = []     
        episode_lengths = []
        last_log = time.time()
        last_sync = time.time()
        reward_component_sums = {
            "reward_total": 0.0,
            "reward_progress": 0.0,
            "reward_speed": 0.0,
            "reward_side_slip_penalty": 0.0,
            "reward_forward_slip_penalty": 0.0,
            "reward_idle_penalty": 0.0,
            "reward_terminal_bonus": 0.0,
            "reward_terminal_penalty": 0.0,
            "speed": 0.0,
        }
        reward_component_count = 0
        
        while run_flag.value == 1:
            if steps < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                action_tensor = actor.act(obs, test=False)
                action = action_tensor.cpu().numpy().flatten()
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if steps % 100 == 0 and steps > 0 and os.path.exists(f"{model_path}.zip"):
                try:
                    old_sum = sum(p.sum().item() for p in actor.sac_model.policy.parameters())
                    actor.load(model_path)
                    new_sum = sum(p.sum().item() for p in actor.sac_model.policy.parameters())
                    
                    sync_elapsed = time.time() - last_sync
                    diff = abs(new_sum - old_sum)
                    
                    if diff > 0:
                        print(f"[COLLECTOR] Sync @ {steps} | Weight delta: {diff:.4f} (Success) | Time: {sync_elapsed:.3f}s")
                    else:
                        print(f"[COLLECTOR] Sync @ {steps} | Weight delta: 0.0000 (WARNING: Weights are identical!)")
                        
                    last_sync = time.time()
                except Exception as e:
                    print(f"[COLLECTOR] Failed to sync weights: {e}")
            
            current_ep_reward += reward  
            current_ep_length += 1

            for key in reward_component_sums:
                reward_component_sums[key] += float(info.get(key, 0.0))
            reward_component_count += 1
            
            experience = (obs, action, reward, next_obs, done)
            
            try:
                experience_queue.put(experience, timeout=0.5)
                steps += 1
                if steps >= MAX_STEPS:
                    print(f"[COLLECTOR] Reached limit MAX_STEPS ({MAX_STEPS}). Finishing training...")
                    run_flag.value = 0
                    break
                
                if done:
                    episode_returns.append(current_ep_reward)
                    episode_lengths.append(current_ep_length)
                    print(f"\n[EPISODE] #{len(episode_returns)} FINISHED | Return: {current_ep_reward:+.2f} | Steps: {current_ep_length} | Status: {'WIN' if current_ep_reward > 50 else 'FAIL'}\n")
                    current_ep_reward = 0.0  
                    current_ep_length = 0
                
                if time.time() - last_log >= 5.0 and len(episode_returns) > 0:
                    avg_return = np.mean(episode_returns[-50:]) if len(episode_returns) >= 1 else 0
                    avg_length = np.mean(episode_lengths[-50:]) if len(episode_lengths) >= 1 else 0
                    max_return = np.max(episode_returns[-50:]) if len(episode_returns) >= 1 else 0
                    print(
                        f"[COLLECTOR] Steps: {steps:6d}"
                        f"| Avg Ep Return: {avg_return:.2f} | Avg Len: {avg_length:.1f} | Max: {max_return:.2f}"
                    )
                    tb_writer.add_scalar('Collector/Episode_Return_Mean_50', avg_return, len(episode_returns))
                    tb_writer.add_scalar('Collector/Episode_Length_Mean_50', avg_length, len(episode_returns))
                    tb_writer.add_scalar('Collector/Episode_Return_Max_50', max_return, len(episode_returns))
                    tb_writer.add_scalar('Collector/Total_Steps', steps, steps)

                    if reward_component_count > 0:
                        tb_writer.add_scalar('CollectorReward/Total_Mean', reward_component_sums['reward_total'] / reward_component_count, steps)
                        tb_writer.add_scalar('CollectorReward/Progress_Mean', reward_component_sums['reward_progress'] / reward_component_count, steps)
                        tb_writer.add_scalar('CollectorReward/Speed_Mean', reward_component_sums['reward_speed'] / reward_component_count, steps)
                        tb_writer.add_scalar('CollectorReward/SideSlipPenalty_Mean', reward_component_sums['reward_side_slip_penalty'] / reward_component_count, steps)
                        tb_writer.add_scalar('CollectorReward/ForwardSlipPenalty_Mean', reward_component_sums['reward_forward_slip_penalty'] / reward_component_count, steps)
                        tb_writer.add_scalar('CollectorReward/IdlePenalty_Mean', reward_component_sums['reward_idle_penalty'] / reward_component_count, steps)
                        tb_writer.add_scalar('CollectorReward/TerminalBonus_Mean', reward_component_sums['reward_terminal_bonus'] / reward_component_count, steps)
                        tb_writer.add_scalar('CollectorReward/TerminalPenalty_Mean', reward_component_sums['reward_terminal_penalty'] / reward_component_count, steps)
                        tb_writer.add_scalar('CollectorState/Speed_Mean', reward_component_sums['speed'] / reward_component_count, steps)

                        for key in reward_component_sums:
                            reward_component_sums[key] = 0.0
                        reward_component_count = 0

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

def learner_worker(experience_queue, run_flag, logs_dir, learner_buffer_size):
    print("[TRAINER] Starting...")
    
    learner_log_dir = os.path.join(logs_dir, "learner")
    os.makedirs(learner_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=learner_log_dir)
    
    try:
        observation_space = spaces.Dict({
            "vision": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            "telemetry": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        })
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        model_path = os.path.join(MODEL_CHECKPOINT_DIR, MODEL_NAME)
        
        if RESUME_TRAINING:
            if os.path.exists(f"{model_path}.zip"):
                print(f"[TRAINER] Resuming from target model: {model_path}")
                training_agent = TmrlSacTrainingAgent(
                    observation_space,
                    action_space,
                    device="cuda",
                    model_path=model_path,
                    buffer_size=MEMORY_SIZE,
                    model_size=MODEL_SIZE,
                )
            elif os.path.exists(f"{PRETRAINED_MODEL_PATH}.zip"):
                print(f"[TRAINER] Loading pretrained model: {PRETRAINED_MODEL_PATH}")
                training_agent = TmrlSacTrainingAgent(
                    observation_space,
                    action_space,
                    device="cuda",
                    model_path=PRETRAINED_MODEL_PATH,
                    buffer_size=MEMORY_SIZE,
                    model_size=MODEL_SIZE,
                )
            else:
                print("[TRAINER] No target or pretrained model found. Creating new training agent")
                training_agent = TmrlSacTrainingAgent(
                    observation_space,
                    action_space,
                    device="cuda",
                    buffer_size=MEMORY_SIZE,
                    model_size=MODEL_SIZE,
                )
        else:
            print(f"[TRAINER] Creating new training agent")
            training_agent = TmrlSacTrainingAgent(
                observation_space,
                action_space,
                device="cuda",
                buffer_size=MEMORY_SIZE,
                model_size=MODEL_SIZE,
            )
        
        print("[TRAINER] Ready")
        
        memory = deque(maxlen=MEMORY_SIZE)
        batches_done = 0
        env_steps_seen = 0
        next_checkpoint_step = CHECKPOINT_STEPS
        last_log_time = time.time()
        
        while run_flag.value == 1:
            try:
                experiences_batch = []
                try:
                    for _ in range(BATCH_SIZE):
                        exp = experience_queue.get(timeout=0.1)
                        experiences_batch.append(exp)
                        memory.append(exp)
                except mp.queues.Empty:
                    pass

                env_steps_seen += len(experiences_batch)
                
                if len(memory) >= TRAIN_BATCH_THRESHOLD:
                    try:
                        if len(experiences_batch) >= TRAIN_BATCH_THRESHOLD:
                            train_batch = experiences_batch
                        else:
                            sample_size = min(BATCH_SIZE, len(memory))
                            train_batch = random.sample(memory, sample_size)

                        training_agent.train(train_batch)
                        batches_done += 1
                        learner_buffer_size.value = training_agent.actor_module.sac_model.replay_buffer.size()
                        
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
                        
                        if batches_done % 50 == 0:
                            try:
                                training_agent.save(model_path)
                                if os.path.exists(f"{model_path}.zip"):
                                    print(f"[TRAINER] CHECKPOINT batch {batches_done} saved successfully")
                                else:
                                    print(f"[TRAINER] ERROR: Model file not created at {model_path}.zip")
                            except Exception as save_err:
                                print(f"[TRAINER] SAVE ERROR at batch {batches_done}: {save_err}")
                                import traceback
                                traceback.print_exc()

                        while env_steps_seen >= next_checkpoint_step:
                            try:
                                checkpoint_model_path = f"{model_path}_steps_{next_checkpoint_step}"
                                training_agent.save(checkpoint_model_path)
                                if os.path.exists(f"{checkpoint_model_path}.zip"):
                                    print(f"[TRAINER] STEP CHECKPOINT saved: {checkpoint_model_path}.zip")
                                else:
                                    print(f"[TRAINER] ERROR: Step checkpoint not created at {checkpoint_model_path}.zip")
                            except Exception as save_err:
                                print(f"[TRAINER] STEP CHECKPOINT SAVE ERROR at {next_checkpoint_step}: {save_err}")
                                import traceback
                                traceback.print_exc()

                            next_checkpoint_step += CHECKPOINT_STEPS
                    
                    except Exception as e:
                        print(f"[TRAINER] Train error at batch {batches_done}: {e}")
                        import traceback
                        traceback.print_exc()
                
            except Exception as e:
                print(f"[TRAINER] ERROR: {e}")
                break
        
        try:
            training_agent.save(model_path)
            if os.path.exists(f"{model_path}.zip"):
                print(f"[TRAINER] Final checkpoint saved at {model_path}.zip after {batches_done} batches")
            else:
                print(f"[TRAINER] ERROR: Final model file not created at {model_path}.zip")
        except Exception as final_err:
            print(f"[TRAINER] FINAL SAVE ERROR: {final_err}")
            import traceback
            traceback.print_exc()

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
    model_cfg = MODEL_SIZE_CONFIG.get(MODEL_SIZE, {})
    print(
        f"Model Size: {MODEL_SIZE} | Features: {model_cfg.get('features_dim', 'n/a')} "
        f"| Net: {model_cfg.get('net_arch', 'n/a')}"
    )
    print(f"Gamma: {GAMMA} | Tau: {TAU} | Auto Entropy: {AUTO_ENTROPY}")
    print(f"Step checkpoint cadence: every {CHECKPOINT_STEPS} env steps")
    print(f"Resume: {RESUME_TRAINING} | Model: {MODEL_NAME}")
    print("="*80 + "\n")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_logs_dir = os.path.join("./logs", MODEL_NAME, timestamp)
    os.makedirs(versioned_logs_dir, exist_ok=True)
    print(f"[MAIN] Logs saved to: {versioned_logs_dir}\n")
    
    mp.set_start_method('spawn', force=True)
    
    exp_queue = mp.Queue(maxsize=2000)
    run_flag = mp.Value('i', 1)
    learner_buffer_size = mp.Value('i', 0)
    
    p_learner = mp.Process(
        target=learner_worker,
        args=(exp_queue, run_flag, versioned_logs_dir, learner_buffer_size),
        daemon=False,
    )
    p_collector = mp.Process(
        target=collector_worker,
        args=(exp_queue, run_flag, versioned_logs_dir, learner_buffer_size),
        daemon=False,
    )
    
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
