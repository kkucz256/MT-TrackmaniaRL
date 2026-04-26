import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import numpy as np
import pandas as pd
import torch
import cv2
import csv
import datetime as dt

from trackmania_env import TrackmaniaEnv
from tmrl_sac_agent import TmrlSacActorModule

MATRIX_TEST_EPISODES = 1 

MODELS_TO_TEST = [
    # Baseline
    ("./models/base/model_track_01_Small_400000steps_buf50000_steps_400000.zip", "Small", "Baseline_Small"),
    ("./models/base/model_track_01_Base_400000steps_buf50000_steps_400000.zip", "Base", "Baseline_Base"),
    ("./models/base/model_track_01_Large_400000steps_buf50000_steps_380000.zip", "Large", "Baseline_Large"),
    
    # Forgetting (Track 01 -> 06)
    ("./models/track_01_on_06/model_forgetting_01_to_06_Small_400000steps_steps_400000.zip", "Small", "Forgetting_Small"),
    ("./models/track_01_on_06/model_forgetting_01_to_06_Base_400000steps_steps_400000.zip", "Base", "Forgetting_Base"),
    ("./models/track_01_on_06/model_forgetting_01_to_06_Large_400000steps_steps_380000.zip", "Large", "Forgetting_Large"),
    
    # Interleaved / Mix
    ("./models/model_mix_track_01_06Small_800000steps_buf50000_steps_400000.zip", "Small", "Mix_Small"),
    ("./models/model_mix_track_01_06Base_800000steps_buf50000_steps_400000.zip", "Base", "Mix_Base"),
    ("./models/model_mix_track_01_06Large_800000steps_buf50000_steps_400000.zip", "Large", "Mix_Large"),
]

DYNAMIC_TEST_EPISODES = 50
DYNAMIC_FORGETTING_DIR = "./models/track_01_on_06/"
DYNAMIC_MODELS = ["Small", "Base", "Large"]
CHECKPOINT_STEPS = list(range(20000, 400001, 20000))

def save_raw_data(raw_data_list, filename):
    if not raw_data_list:
        return
    df = pd.DataFrame(raw_data_list)
    df.to_csv(filename, index=False)

def _get_loaded_model_size(actor):
    try:
        extractor = actor.sac_model.policy.features_extractor
        return getattr(extractor, "model_size", None)
    except Exception:
        return None

def _evaluate_single_model(env, model_path, model_size, test_name, episodes, step_checkpoint=None):
    print(f"\n--- [EVAL] Testowanie: {test_name} ({model_path}) ---")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] BRAK PLIKU: {model_path}. Pomijam.")
        return None, []

    try:
        actor = TmrlSacActorModule(
            env.observation_space, 
            env.action_space, 
            device="cuda", 
            model_path=model_path, 
            buffer_size=1000,
            model_size=model_size
        )
    except Exception as e:
        print(f"[ERROR] Błąd ładowania modelu: {e}")
        return None, []

    results = []
    
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        steps = 0
        ep_speed_sum = 0.0
        ep_reward_sum = 0.0
        status = "TIMEOUT"
        
        ep_start_time = time.time()
        
        while not done:
            action_tensor = actor.act(obs, test=True)
            action = action_tensor.cpu().numpy().flatten()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            steps += 1
            ep_speed_sum += info.get('speed', 0.0)
            ep_reward_sum += reward
            
            if info.get('reward_terminal_bonus', 0.0) >= 50.0:
                status = "WIN"
            elif info.get('reward_terminal_penalty', 0.0) <= -20.0:
                status = "CRASH"
                
        ep_end_time = time.time()
        wall_time_sec = ep_end_time - ep_start_time
        
        expected_game_time_sec = steps * 0.05 
        actual_hz = steps / wall_time_sec if wall_time_sec > 0 else 0
        avg_speed = ep_speed_sum / steps if steps > 0 else 0.0
        
        ep_data = {
            "Test_Name": test_name,
            "Model_Size": model_size,
            "Checkpoint_Step": step_checkpoint if step_checkpoint else "N/A",
            "Episode": ep,
            "Status": status,
            "Steps": steps,
            "Expected_InGame_Time_Sec": expected_game_time_sec,
            "Wall_Time_Sec": wall_time_sec,
            "Actual_Hz": actual_hz,
            "Avg_Speed": avg_speed,
            "Total_Reward": ep_reward_sum
        }
        results.append(ep_data)
        
        if ep % 5 == 0 or ep == episodes:
            print(f"  > Postęp: {ep}/{episodes} | Ostatni: {status} (R: {ep_reward_sum:.1f}, Kroki: {steps}, Czas gry: {expected_game_time_sec:.2f}s, Hz: {actual_hz:.1f})")

    wins = [r for r in results if r["Status"] == "WIN"]
    success_rate = (len(wins) / episodes) * 100
    
    avg_reward = np.mean([r["Total_Reward"] for r in results])
    std_reward = np.std([r["Total_Reward"] for r in results])
    
    avg_speed_all = np.mean([r["Avg_Speed"] for r in results])
    
    avg_steps = np.mean([r["Steps"] for r in results])
    avg_game_time = np.mean([r["Expected_InGame_Time_Sec"] for r in results])
    avg_wall_time = np.mean([r["Wall_Time_Sec"] for r in results])
    avg_hz = np.mean([r["Actual_Hz"] for r in results])
    
    avg_win_steps = np.mean([r["Steps"] for r in wins]) if wins else 0.0
    avg_win_time = np.mean([r["Expected_InGame_Time_Sec"] for r in wins]) if wins else 0.0

    print(f"--- Wynik {test_name}: SR={success_rate:.1f}%, Średni Czas Gry: {avg_game_time:.2f}s, Wydajność: {avg_hz:.1f} Hz ---")
    
    metrics_summary = {
        "Test_Name": test_name,
        "Success_Rate_%": success_rate,
        "Mean_Reward": avg_reward,
        "Std_Reward": std_reward,
        "Mean_Speed_Overall": avg_speed_all,
        "Mean_Steps": avg_steps,
        "Mean_Win_Steps": avg_win_steps,
        "Mean_Expected_InGame_Time": avg_game_time,
        "Mean_Expected_Win_Time": avg_win_time,
        "Mean_Wall_Computation_Time": avg_wall_time,
        "Mean_Agent_Hz": avg_hz
    }
    
    return metrics_summary, results

def run_matrix_evaluation():
    print("\n" + "="*80)
    print(f"ROZPOCZYNAM EWALUACJĘ MACIERZOWĄ ({MATRIX_TEST_EPISODES} epizodów na model)")
    print("="*80)
    
    env = TrackmaniaEnv()
    
    print("[EVAL] Oczekiwanie na Trackmanię (F3 -> Reload)...")
    while True:
        _, tele = env.pipeline.get_state()
        if tele:
            break
        time.sleep(1.0)
        
    print("[EVAL] Połączono! Startuję maraton.\n")
    
    all_summary_metrics = []
    all_raw_data = []
    
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_csv_file = f"./eval/summary_matrix_{timestamp}.csv"
    raw_csv_file = f"./eval/raw_matrix_{timestamp}.csv"
    
    for path, size, name in MODELS_TO_TEST:
        metrics, raw_episodes = _evaluate_single_model(env, path, size, name, MATRIX_TEST_EPISODES)
        
        if metrics and raw_episodes:
            all_summary_metrics.append(metrics)
            all_raw_data.extend(raw_episodes)
            
            keys = all_summary_metrics[0].keys()
            with open(summary_csv_file, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(all_summary_metrics)
            save_raw_data(all_raw_data, raw_csv_file)
            
    env.close()
    print(f"\n[SUKCES] Macierz zakończona.")
    print(f" -> Podsumowanie: {summary_csv_file}")
    print(f" -> Surowe dane: {raw_csv_file}")

def run_dynamic_forgetting_evaluation():
    print("\n" + "="*80)
    print(f"ROZPOCZYNAM EWALUACJĘ DYNAMIKI ZAPOMINANIA ({DYNAMIC_TEST_EPISODES} epizodów na krok)")
    print("="*80)
    
    env = TrackmaniaEnv()
    
    print("[EVAL] Oczekiwanie na Trackmanię (F3 -> Reload)...")
    while True:
        _, tele = env.pipeline.get_state()
        if tele:
            break
        time.sleep(1.0)
        
    all_summary_metrics = []
    all_raw_data = []
    
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_csv_file = f"./eval/summary_dynamic_{timestamp}.csv"
    raw_csv_file = f"./eval/raw_dynamic_{timestamp}.csv"
    
    for size in DYNAMIC_MODELS:
        print(f"\n\n>>> BADANIE ROZPADU: ARCHITEKTURA {size.upper()} <<<")
        for step in CHECKPOINT_STEPS:
            if size == "Large" and step == 400000:
                filename = f"model_forgetting_01_to_06_{size}_400000steps.zip"
                print(f"[INFO] Używam ostatecznego pliku (fallback) dla Large 400k.")
            else:
                filename = f"model_forgetting_01_to_06_{size}_400000steps_steps_{step}.zip"
                
            path = os.path.join(DYNAMIC_FORGETTING_DIR, filename)
            name = f"Forgetting_{size}_{step//1000}k"
            
            if not os.path.exists(path):
                alt_path = os.path.join(DYNAMIC_FORGETTING_DIR, f"model_forgetting_01_to_06_{size}_400000steps.zip")
                if step == 400000 and os.path.exists(alt_path):
                     path = alt_path
                     print(f"[WARN] Brak: {filename}. Awaryjnie ładuję: {alt_path}")
                else:
                     print(f"[ERROR] BRAK PLIKU: {path}. Pomijam krok {step}.")
                     continue
            
            metrics, raw_episodes = _evaluate_single_model(env, path, size, name, DYNAMIC_TEST_EPISODES, step_checkpoint=step)
            
            if metrics and raw_episodes:
                metrics["Model_Size"] = size
                metrics["Checkpoint_Step"] = step 
                all_summary_metrics.append(metrics)
                all_raw_data.extend(raw_episodes)
                
                keys = all_summary_metrics[0].keys()
                with open(summary_csv_file, 'w', newline='') as output_file:
                    dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(all_summary_metrics)
                save_raw_data(all_raw_data, raw_csv_file)
                    
    env.close()
    print(f"\n[SUKCES] Zakończono maraton dynamiki.")
    print(f" -> Podsumowanie: {summary_csv_file}")
    print(f" -> Surowe dane: {raw_csv_file}")

if __name__ == "__main__":
    print("=== AUTOMATYCZNY SYSTEM EWALUACJI TMRL ===")
    print("1: Macierz Ewaluacji (Rozdział 4 - Główna Tabela | 9 modeli)")
    print("2: Dynamika Zapominania (Rozdział 4 - Wykres Amnezji | ~60 modeli)")
    
    choice = input("\nWybierz test (1/2): ")
    
    try:
        if choice == '1':
            print("\n[UWAGA] Pamiętaj o manualnej zmianie trasy w grze po skończeniu!")
            run_matrix_evaluation()
        elif choice == '2':
            print("\n[UWAGA] Upewnij się, że w grze załadowana jest TRASA 01!")
            run_dynamic_forgetting_evaluation()
        else:
            print("Niepoprawny wybór. Koniec działania.")
    except KeyboardInterrupt:
        print("\n[EVAL] Przerwano ręcznie proces ewaluacji.")