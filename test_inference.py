"""
TEST INFERENCE - Testowanie wytrenowanego modelu DQN
Uruchamia grę bez uczenia, tylko obserwuje i testuje politykę
"""
import torch
import numpy as np
import time
from trackmania_env import TrackmaniaEnv
from model import TrackmaniaNet

def test_inference(num_episodes=5, max_steps=1000):
    print("\n" + "="*70)
    print("TEST INFERENCE - WYTRENOWANY MODEL DQN")
    print("="*70)
    
    env = TrackmaniaEnv()
    model = TrackmaniaNet(env.observation_space, env.action_space).cuda()
    model.eval()
    
    try:
        model.load_state_dict(torch.load("model_nd.pt", weights_only=True))
        print("Model załadowany z model_weights.pt")
    except FileNotFoundError:
        print("BŁĄD: model_weights.pt nie znaleziony!")
        print("Uruchom najpierw trening: python decoupled_learner.py")
        return
    except Exception as e:
        print(f"Błąd przy ładowaniu modelu: {e}")
        return
    
    print("="*70 + "\n")
    
    total_reward = 0.0
    episode_rewards = []
    
    for episode in range(num_episodes):
        print(f"[EPIZOD {episode+1}/{num_episodes}] Startuje...")
        
        obs_dict, _ = env.reset()
        episode_reward = 0.0
        steps = 0
        
        for step in range(max_steps):
            with torch.no_grad():
                v = torch.tensor(obs_dict["vision"]).unsqueeze(0).cuda()
                t = torch.tensor(obs_dict["telemetry"]).unsqueeze(0).cuda()
                q_values = model(v, t)
                action = int(torch.argmax(q_values, dim=1).item())
            
            next_obs_dict, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            if step % 100 == 0:
                print(f"  Step {step:3d}/{max_steps} | Action: {action} | Reward: {reward:7.3f} | Total: {episode_reward:8.3f}")
            
            obs_dict = next_obs_dict
            
            if done:
                print(f" RESET! (Step {steps})")
                obs_dict, _ = env.reset()
        
        episode_rewards.append(episode_reward)
        total_reward += episode_reward
        
        print(f"[KONIEC EPIZODU {episode+1}] Nagroda: {episode_reward:.3f}")
        print()
    
    avg_reward = total_reward / num_episodes
    min_reward = min(episode_rewards)
    max_reward = max(episode_rewards)
    
    print("="*70)
    print("PODSUMOWANIE TESTU")
    print("="*70)
    print(f"Epizody: {num_episodes}")
    print(f"Średnia nagroda: {avg_reward:.3f}")
    print(f"Min nagroda: {min_reward:.3f}")
    print(f"Max nagroda: {max_reward:.3f}")
    print(f"Nagrody po epizodzie: {[f'{r:.2f}' for r in episode_rewards]}")
    print("="*70 + "\n")
    
    env.close()

if __name__ == "__main__":
    import sys
    
    num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    
    test_inference(num_episodes=num_episodes)
