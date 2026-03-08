import gymnasium as gym
from gymnasium import spaces
import numpy as np
import vgamepad as vg
import time
import cv2
import pandas as pd
from scipy.spatial import KDTree
import os

from pipeline import TrackmaniaPipeline 

class TrackmaniaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.pipeline = TrackmaniaPipeline()
        
        self.gamepad = vg.VX360Gamepad()
        
        self.last_pos = None
        self.pause_counter = 0
        self.pause_threshold = 10
        
        try:
            track_points_path = os.path.join(os.getcwd(), "trackpoints", "track_points_training01.csv")
            df = pd.read_csv(track_points_path, header=None)
            self.track_points = df.values
            self.kdtree = KDTree(self.track_points)
            self.last_track_index = 0
        except Exception as e:
            print(f"[TrackmaniaEnv] Warning: Could not load track points from {track_points_path}: {e}")
            self.track_points = np.zeros((1, 3))
            self.kdtree = KDTree(self.track_points)
            self.last_track_index = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            "vision": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            "telemetry": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        })

    def _calculate_reward(self, current_pos, current_speed):
        dist, index = self.kdtree.query(current_pos)
        
        progress = index - self.last_track_index
        if progress > 0:
            progress_reward = 15.0 * progress
            self.last_track_index = index
        else:
            progress_reward = -0.05
        
        if dist < 1.5:
            track_bonus = 0.2
        elif dist < 4.0:
            track_bonus = 0.1
        elif dist < 10.0:
            track_bonus = 0.0
        else:
            track_bonus = -0.2
        
        # MUCH STRONGER speed bonus - encourage forward movement
        speed_bonus = 0.0
        if 25.0 <= current_speed <= 200.0:
            speed_bonus = 1.5
        elif 10.0 <= current_speed < 25.0:
            speed_bonus = 0.5
        elif current_speed < 5.0:
            speed_bonus = -1.0
            
        total_reward = progress_reward + track_bonus + speed_bonus
        
        return float(total_reward), dist

    def step(self, action):
        # Handle continuous action: [steering, throttle/brake]
        # steering: [-1, 1] left to right
        # throttle/brake: [-1, 1] where >0 is gas, <0 is brake
        
        if isinstance(action, np.ndarray):
            steering = float(np.clip(action[0], -1.0, 1.0))
            throttle_brake = float(np.clip(action[1], -1.0, 1.0))
        else:
            steering = 0.0
            throttle_brake = 0.0
        
        # Throttle: if positive
        throttle = max(0.0, throttle_brake)
        # Brake: if negative (flip to positive)
        brake = max(0.0, -throttle_brake)
        
        # Map to gamepad
        x_joystick = int(steering * 32767)
        self.gamepad.left_joystick(x_value=x_joystick, y_value=0)
        self.gamepad.right_trigger(value=int(throttle * 255))
        self.gamepad.left_trigger(value=int(brake * 255))
        self.gamepad.update()

        time.sleep(0.1)

        frame, tele = self.pipeline.get_state()
        state = self._format_state(frame, tele)

        current_pos = np.array([tele.get('pos_x', 0), tele.get('pos_y', 0), tele.get('pos_z', 0)])
        reward, distance_to_line = self._calculate_reward(current_pos, tele.get('speed', 0))

        terminated = False
        
        is_finished = tele.get('is_finished', False)
        
        if is_finished:
            print("[ENV] FINISH LINE - REWARD: +100.0")
            reward = 100.0
            terminated = True
        
        elif distance_to_line > 10.0:
            print(f"[ENV] OFF-TRACK (dist: {distance_to_line:.2f}) - PENALTY: -0.3")
            reward = -0.3
            terminated = True
        
        speed = tele.get('speed', 0)
        if speed < 1.0:
            if self.last_pos is not None:
                dist_from_last = np.linalg.norm(current_pos - self.last_pos)
                if dist_from_last < 0.1:
                    self.pause_counter += 1
                    if self.pause_counter > self.pause_threshold:
                        print(f"[ENV] PAUSE DETECTED (pause_counter: {self.pause_counter}) - PENALTY: -0.2")
                        reward = -0.2
                        terminated = True
                        self.pause_counter = 0
                else:
                    self.pause_counter = 0
            self.last_pos = current_pos.copy()
        else:
            self.pause_counter = 0
            self.last_pos = current_pos.copy()

        return state, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.last_track_index = 0
        self.pause_counter = 0
        
        print("\n" + "="*60)
        print("[RESET] New episode - resetting environment and respawning agent...")
        print("="*60 + "\n")
        
        for _ in range(2):
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            self.gamepad.update()
            time.sleep(0.1)
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            self.gamepad.update()
            time.sleep(0.3)
        
        time.sleep(1.5)
        
        if hasattr(self.pipeline, 'flush'):
            self.pipeline.flush()
        
        frame, tele = self.pipeline.get_state()
        
        if tele.get('is_finished', False):
             print("[ENV WARN] Agent respawned with is_finished flag. Bad UI?")
             
        return self._format_state(frame, tele), {}

    def _format_state(self, frame, tele):
        tele_vector = np.array([
            tele.get('speed', 0.0),
            tele.get('pos_x', 0.0), tele.get('pos_y', 0.0), tele.get('pos_z', 0.0),
            tele.get('vel_x', 0.0), tele.get('vel_y', 0.0), tele.get('vel_z', 0.0)
        ], dtype=np.float32)
        
        if frame is None:
            frame = np.zeros((128, 128, 3), dtype=np.uint8)
            
        return {"vision": frame, "telemetry": tele_vector}

    def close(self):
        self.pipeline.stop()
        self.gamepad.reset()
        self.gamepad.update()

if __name__ == "__main__":
    env = TrackmaniaEnv()
    obs, info = env.reset()
    
    print("Starting test run with random actions (Centerline Reward Mode)...")
    try:
        for step in range(200):
            random_action = env.action_space.sample() 
            obs, reward, terminated, truncated, info = env.step(random_action)
            
            cv2.imshow("Agent Vision", obs["vision"])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            print(f"Step: {step:3} | Reward: {reward:6.2f} | Progress: {env.last_track_index}")
            
            if terminated:
                print("--- OFF TRACK - RESET ---")
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        cv2.destroyAllWindows()