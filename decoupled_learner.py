#!/usr/bin/env python3
"""
DECOUPLED LEARNER DLA TRACKMANII
3 PROCESY: Kolektor + Trener + Dashboard
"""
import multiprocessing as mp
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import cv2
import datetime
import sys
import csv
from collections import deque

from trackmania_env import TrackmaniaEnv 
from model import TrackmaniaNet

# === HIPERPARAMETRY ===
GAMMA = 0.95
LR = 5e-4
BATCH_SIZE = 32
MEMORY_SIZE = 5000
TARGET_UPDATE = 5
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 10000

print(f"""
{'='*70}
TRENER DQN DLA TRACKMANII - DECOUPLED LEARNER
{'='*70}
Hiperparametry:
- Learning Rate: {LR}
- Gamma: {GAMMA}
- Batch Size: {BATCH_SIZE}
- Memory Size: {MEMORY_SIZE}
- Target Update: {TARGET_UPDATE}
- EPS Decay: {EPS_DECAY}
{'='*70}
""")

def collector_worker(experience_queue, run_flag):
    """
    PROCES 1: KOLEKTOR (ACTOR)
    Zbiera doświadczenia i wysyła je do Trenera
    """
    print("\n[KOLEKTOR] Inicjalizacja...")
    
    collector_csv = f"collector_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    collector_csv_file = None
    collector_csv_writer = None
    env = None
    
    try:
        env = TrackmaniaEnv()
        model = TrackmaniaNet(env.observation_space, env.action_space).cuda()
        model.eval()
        
        obs_dict, _ = env.reset()
        steps = 0
        last_reward_sum = 0.0
        last_reward_log = time.time()
        
        print("[KOLEKTOR] Gotowy, zaczynam zbierać doświadczenia...\n")
        
        while run_flag.value == 1:
            if steps > 0 and steps % 150 == 0 and os.path.exists("model_weights.pt"):
                try:
                    model.load_state_dict(torch.load("model_weights.pt", weights_only=True))
                except Exception as e:
                    print(f"[KOLEKTOR] Błąd ładowania: {e}")

            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps / EPS_DECAY)
            
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    v = torch.tensor(obs_dict["vision"]).unsqueeze(0).cuda()
                    t = torch.tensor(obs_dict["telemetry"]).unsqueeze(0).cuda()
                    q_values = model(v, t)
                    action = int(torch.argmax(q_values, dim=1).item())
            
            next_obs_dict, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            experience = (
                obs_dict["vision"],
                obs_dict["telemetry"],
                action,
                reward,
                next_obs_dict["vision"],
                next_obs_dict["telemetry"],
                done
            )
            
            try:
                experience_queue.put(experience, timeout=0.5)
                last_reward_sum += reward
                steps += 1
                
                if time.time() - last_reward_log >= 5.0:
                    avg_rew = last_reward_sum / 50 if steps > 0 else 0
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    print(f"[KOLEKTOR] Steps: {steps:6d} | Avg Reward (50): {avg_rew:8.4f} | Eps: {epsilon:.4f}")
                    
                    try:
                        if collector_csv_file is None:
                            collector_csv_file = open(collector_csv, 'a', newline='')
                            collector_csv_writer = csv.writer(collector_csv_file)
                            collector_csv_writer.writerow(['Timestamp', 'Steps', 'Avg_Reward_50', 'Epsilon', 'Done_Count'])
                        
                        done_count = sum(1 for exp in [experience] if exp[6])
                        collector_csv_writer.writerow([timestamp, steps, f"{avg_rew:.4f}", f"{epsilon:.4f}", done_count])
                        collector_csv_file.flush()
                    except Exception as e:
                        print(f"[KOLEKTOR CSV] Błąd: {e}")
                    
                    last_reward_sum = 0.0
                    last_reward_log = time.time()
                    
            except mp.queues.Full:
                pass
            
            obs_dict = next_obs_dict
            if done:
                obs_dict, _ = env.reset()

    except Exception as e:
        print(f"[KOLEKTOR] BŁĄD: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if collector_csv_file:
            collector_csv_file.close()
            print(f"[KOLEKTOR CSV] Logi zapisane do: {collector_csv}")
        if env:
            env.close()
        print("[KOLEKTOR] Zamknięty.")

def learner_worker(experience_queue, run_flag):
    """
    PROCES 2: TRENER (LEARNER)
    Trenuje model na doświadczeniach z Kolektora
    """
    print("[TRENER] Inicjalizacja sieci DQN...\n")
    
    try:
        tmp_env = TrackmaniaEnv()
        policy_net = TrackmaniaNet(tmp_env.observation_space, tmp_env.action_space).cuda()
        target_net = TrackmaniaNet(tmp_env.observation_space, tmp_env.action_space).cuda()
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        tmp_env.close()

        optimizer = optim.Adam(policy_net.parameters(), lr=LR)
        criterion = nn.SmoothL1Loss()
        
        memory = deque(maxlen=MEMORY_SIZE)
        batches_done = 0
        last_log_time = time.time()
        exp_count = 0
        
        csv_file = f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_writer = None
        csv_file_handle = None
        
        print("[TRENER] Czekam na doświadczenia...\n")
        
        while run_flag.value == 1:
            try:
                exp = experience_queue.get(timeout=2.0)
                memory.append(exp)
                exp_count += 1
                
                if len(memory) >= BATCH_SIZE:
                    batch = random.sample(memory, BATCH_SIZE)
                    
                    v_b = torch.tensor(np.array([m[0] for m in batch])).cuda()
                    t_b = torch.tensor(np.array([m[1] for m in batch])).cuda()
                    a_b = torch.tensor(np.array([m[2] for m in batch])).long().cuda()
                    r_b = torch.tensor(np.array([m[3] for m in batch])).float().cuda()
                    v_n = torch.tensor(np.array([m[4] for m in batch])).cuda()
                    t_n = torch.tensor(np.array([m[5] for m in batch])).cuda()
                    d_b = torch.tensor(np.array([m[6] for m in batch])).float().cuda()

                    with torch.no_grad():
                        next_q_values = target_net(v_n, t_n)
                        max_next_q = next_q_values.max(dim=1)[0]
                        target_q = r_b + (1 - d_b) * GAMMA * max_next_q

                    q_pred = policy_net(v_b, t_b)
                    current_q = q_pred.gather(1, a_b.unsqueeze(1)).squeeze(1)
                    loss = criterion(current_q, target_q)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    batches_done += 1
                    
                    if time.time() - last_log_time >= 2.0:
                        avg_reward = np.mean([exp[3] for exp in list(memory)[-min(100, len(memory)):]])
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        print(f"[TRENER] Batch: {batches_done:5d} | Loss: {loss.item():.4f} | Avg_Reward: {avg_reward:8.4f} | Buffer: {len(memory):5d}")
                        last_log_time = time.time()
                        
                        try:
                            if csv_file_handle is None:
                                csv_file_handle = open(csv_file, 'a', newline='')
                                csv_writer = csv.writer(csv_file_handle)
                                csv_writer.writerow(['Timestamp', 'Batch', 'Loss', 'Avg_Reward', 'Buffer_Size', 'Epsilon'])
                            
                            eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * batches_done / EPS_DECAY)
                            csv_writer.writerow([timestamp, batches_done, f"{loss.item():.4f}", f"{avg_reward:.4f}", len(memory), f"{eps:.4f}"])
                            csv_file_handle.flush()
                        except Exception as e:
                            print(f"[CSV] Błąd: {e}")
                    
                    if batches_done % TARGET_UPDATE == 0:
                        target_net.load_state_dict(policy_net.state_dict())
                        try:
                            torch.save(policy_net.state_dict(), "model_weights.pt")
                            file_size = os.path.getsize("model_weights.pt") / (1024*1024)
                            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                            print(f"\n>>> [CHECKPOINT {timestamp}] Batch: {batches_done} | Loss: {loss.item():.4f} | Size: {file_size:.2f}MB <<<\n")
                            
                            if csv_writer:
                                csv_writer.writerow(['', f'CHECKPOINT BATCH {batches_done}', f"{loss.item():.4f}", '', '', ''])
                                csv_file_handle.flush()
                        except Exception as e:
                            print(f"[BŁĄD ZAPISU] {e}")
                
            except mp.queues.Empty:
                continue
            except Exception as e:
                print(f"[TRENER] BŁĄD: {e}")
                import traceback
                traceback.print_exc()
                break

    except Exception as e:
        print(f"[TRENER] FATALNA INICJALIZACJA: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if csv_file_handle:
            csv_file_handle.close()
            print(f"[CSV] Logi zapisane do: {csv_file}")
        print("[TRENER] Zamknięty.")

def dashboard_worker(experience_queue_peek, run_flag):
    """
    PROCES 3: DASHBOARD
    Wizualna reprezentacja tego co widzi sieć
    """
    print("[DASHBOARD] Uruchamiam wizualizację...\n")
    
    recent_rewards = deque(maxlen=50)
    last_rew_display = time.time()
    
    while run_flag.value == 1:
        try:
            if not experience_queue_peek.empty():
                try:
                    exp = experience_queue_peek.get_nowait()
                    vision = exp[0]
                    reward = exp[3]
                    recent_rewards.append(reward)
                    
                    display_frame = cv2.resize(vision, (512, 512), interpolation=cv2.INTER_NEAREST)
                    
                    color = (0, 255, 0) if reward > 0 else (0, 0, 255)
                    cv2.putText(display_frame, f"Reward: {reward:.3f}", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    avg_rew = np.mean(recent_rewards) if recent_rewards else 0
                    cv2.putText(display_frame, f"Avg(50): {avg_rew:.3f}", (20, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    cv2.imshow("CNN Vision - Agent Eyes", display_frame)
                except:
                    pass
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"[DASHBOARD] Błąd: {e}")
            continue
    
    cv2.destroyAllWindows()
    print("[DASHBOARD] Zamknięty.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    exp_queue = mp.Queue(maxsize=2000)
    exp_queue_peek = mp.Queue(maxsize=100)
    run_flag = mp.Value('i', 1)
    
    p_learner = mp.Process(target=learner_worker, args=(exp_queue, run_flag), daemon=False)
    p_collector = mp.Process(target=collector_worker, args=(exp_queue, run_flag), daemon=False)
    p_dashboard = mp.Process(target=dashboard_worker, args=(exp_queue_peek, run_flag), daemon=True)
    
    try:
        print("[MAIN] Uruchamiam system...\n")
        p_learner.start()
        time.sleep(1)
        p_collector.start()
        time.sleep(1)
        p_dashboard.start()
        
        print("[MAIN] Wszystkie procesy uruchomione. Escape = wyjście.\n")
        print("="*70 + "\n")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n[MAIN] Otrzymano SIGINT. Zamykanie...\n")
    finally:
        run_flag.value = 0
        
        for proc in [p_collector, p_learner, p_dashboard]:
            try:
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.terminate()
            except:
                pass
        
        print("[MAIN] System zamknięty.")
        sys.exit(0)
