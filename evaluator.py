import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import numpy as np
import torch
import cv2

from trackmania_env import TrackmaniaEnv
from tmrl_sac_agent import TmrlSacActorModule


MODEL_PATH = os.path.join(".", "models/base", "model_track_01_Base_400000steps_buf50000_steps_400000.zip")
MODEL_SIZE = "Large"
TEST_EPISODES = 5


def _get_loaded_model_size(actor):
    try:
        extractor = actor.sac_model.policy.features_extractor
        return getattr(extractor, "model_size", None)
    except Exception:
        return None

def run_evaluation():
    print("\n" + "="*80)
    print("URUCHAMIAM STERYLNY TRYB EWALUACJI (DETERMINISTIC=TRUE)")
    print("="*80)
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Nie znaleziono modelu pod ścieżką: {MODEL_PATH}")
        return

    env = TrackmaniaEnv()
    
    print(f"[EVAL] Ładowanie zamrożonych wag z {MODEL_PATH}...")
    
    try:
        actor = TmrlSacActorModule(
            env.observation_space, 
            env.action_space, 
            device="cuda", 
            model_path=MODEL_PATH, 
            buffer_size=1000,
            model_size=MODEL_SIZE
        )
    except TypeError:
        print("[WARN] Actor nie przyjął model_size. Inicjalizacja bez tego parametru.")
        actor = TmrlSacActorModule(
            env.observation_space, 
            env.action_space, 
            device="cuda", 
            model_path=MODEL_PATH, 
            buffer_size=1000
        )

    loaded_model_size = _get_loaded_model_size(actor)
    if MODEL_SIZE is not None and loaded_model_size is not None and MODEL_SIZE != loaded_model_size:
        raise ValueError(
            f"Rozjazd rozmiaru modelu: konfiguracja='{MODEL_SIZE}', model zip='{loaded_model_size}'. "
            "Ustaw poprawny MODEL_SIZE albo MODEL_SIZE = None."
        )

    resolved_model_size = loaded_model_size if loaded_model_size is not None else (MODEL_SIZE or "Unknown")
    print(f"[EVAL] Załadowany model ma rozmiar: {resolved_model_size}")
    print("\n[EVAL] Model załadowany. Oczekiwanie na połączenie z Trackmanią (F3 -> Reload)...")
    while True:
        _, tele = env.pipeline.get_state()
        if tele:
            break
        time.sleep(1.0)
        
    print("[EVAL] Połączono! Zaczynamy pomiary.\n")

    results = []
    
    for ep in range(1, TEST_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        steps = 0
        ep_speed_sum = 0.0
        start_time = time.time()
        
        status = "TIMEOUT/UNKNOWN"
        
        while not done:
            action_tensor = actor.act(obs, test=True)
            action = action_tensor.cpu().numpy().flatten()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            steps += 1
            ep_speed_sum += info.get('speed', 0.0)
            
            if info.get('reward_terminal_bonus', 0.0) >= 50.0:
                status = "WIN"
            elif info.get('reward_terminal_penalty', 0.0) <= -20.0:
                status = "CRASH (Stuck)"
                
        wall_time = time.time() - start_time
        avg_speed = ep_speed_sum / steps if steps > 0 else 0.0
        
        results.append({
            "episode": ep,
            "status": status,
            "steps": steps,
            "time_sec": wall_time,
            "avg_speed": avg_speed
        })
        
        print(f"Epizod {ep:02d}/{TEST_EPISODES} | Status: {status:15} | Czas: {wall_time:05.1f}s | Kroki: {steps:03d} | Śr. Prędkość: {avg_speed:05.1f}")


    wins = [r for r in results if r["status"] == "WIN"]
    success_rate = (len(wins) / TEST_EPISODES) * 100
    
    avg_win_time = np.mean([r["time_sec"] for r in wins]) if wins else 0.0
    avg_win_steps = np.mean([r["steps"] for r in wins]) if wins else 0.0
    
    print("\n" + "="*80)
    print("RAPORT KOŃCOWY EWALUACJI:")
    print("="*80)
    print(f"Model:           {MODEL_PATH} ({resolved_model_size})")
    print(f"Success Rate:    {success_rate:.1f}% ({len(wins)}/{TEST_EPISODES} udanych przejazdów)")
    
    if wins:
        print(f"Średni czas gry: {avg_win_time:.2f} s (Tylko udane przejazdy)")
        print(f"Średnia klatek:  {avg_win_steps:.1f} kroków")
    else:
        print("Średni czas:     BRAK ZAKOŃCZONYCH PRZEJAZDÓW")
    print("="*80 + "\n")

    env.close()

if __name__ == "__main__":
    try:
        run_evaluation()
    except KeyboardInterrupt:
        print("\nPrzerwano ręcznie.")