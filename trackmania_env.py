import gymnasium as gym
from gymnasium import spaces
import numpy as np
import vgamepad as vg
import time
import cv2

from pipeline import TrackmaniaPipeline 

class TrackmaniaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.pipeline = TrackmaniaPipeline()
        
        self.gamepad = vg.VX360Gamepad()
        
        self.last_cps = 0
        self.pause_counter = 0
        self.pause_threshold = 20
        self.frozen_rpm_counter = 0
        self.last_seen_rpm = -1.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            "vision": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            "telemetry": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        })

    def step(self, action):
        if isinstance(action, np.ndarray):
            steering = float(np.clip(action[0], -1.0, 1.0))
            throttle_brake = float(np.clip(action[1], -1.0, 1.0))
        else:
            steering = 0.0
            throttle_brake = 0.0
        
        throttle = max(0.0, throttle_brake)
        brake = max(0.0, -throttle_brake)
        
        x_joystick = int(steering * 32767)
        self.gamepad.left_joystick(x_value=x_joystick, y_value=0)
        self.gamepad.right_trigger(value=int(throttle * 255))
        self.gamepad.left_trigger(value=int(brake * 255))
        self.gamepad.update()

        time.sleep(0.1)

        frame, tele = self.pipeline.get_state()
        speed_for_freeze = float(tele.get('speed', 0.0))
        rpm_for_freeze = float(tele.get('rpm', 0.0))
        
        if speed_for_freeze == 0.0 and rpm_for_freeze == self.last_seen_rpm:
            self.frozen_rpm_counter += 1
        else:
            self.frozen_rpm_counter = 0
        self.last_seen_rpm = rpm_for_freeze

        if tele.get('is_loading', False) or self.frozen_rpm_counter > 15:
            print("\n[ENV] Map change / scoreboard, freezing agent")
            self.gamepad.reset()
            self.gamepad.update()

            while not tele.get('is_loading', False):
                time.sleep(0.5)
                _, tele = self.pipeline.get_state()

            while tele.get('is_loading', False):
                time.sleep(0.5)
                _, tele = self.pipeline.get_state()

            while float(tele.get('rpm', 0.0)) <= 10.0:
                time.sleep(0.1)
                _, tele = self.pipeline.get_state()

            print("[ENV] Engine woke up. Hard reset\n")
            self.frozen_rpm_counter = 0
            frame, tele = self.pipeline.get_state()
            self.last_seen_rpm = float(tele.get('rpm', 0.0))
            state = self._format_state(frame, tele)

            empty_info = {
                "reward_total": 0.0, "reward_progress": 0.0, "reward_speed": 0.0,
                "reward_side_slip_penalty": 0.0, "reward_forward_slip_penalty": 0.0,
                "reward_idle_penalty": 0.0, "reward_terminal_bonus": 0.0,
                "reward_terminal_penalty": 0.0, "speed": 0.0, "cps_passed": 0
            }
            return state, 0.0, True, False, empty_info

        state = self._format_state(frame, tele)

        cps_passed = tele.get('cps_passed', 0)
        speed = float(tele.get('speed', 0.0))
        slip_forward = float(tele.get('slip_forward', 0.0))

        progress_reward = 0.0
        if cps_passed > self.last_cps:
            progress_reward = 50.0
            self.last_cps = cps_passed

        speed_reward = 0.0
        reverse_penalty = 0.0
        
        if slip_forward > 2.0:
            speed_reward = np.clip(slip_forward / 100.0, 0.0, 1.0) * 1.0
        elif slip_forward < -2.0:
            reverse_penalty = -0.5

        side_slip_penalty = 0.0
        idle_penalty = -0.1 if speed < 5.0 else 0.0

        terminal_bonus = 0.0
        terminal_penalty = 0.0
        reward = progress_reward + speed_reward + reverse_penalty + side_slip_penalty + idle_penalty

        terminated = False
        is_finished = tele.get('is_finished', False)
        
        if is_finished:
            print("[ENV] FINISH LINE - REWARD: +100.0")
            terminal_bonus = 100.0
            reward += terminal_bonus
            terminated = True

        if speed < 2.0:
            self.pause_counter += 1
            if self.pause_counter > self.pause_threshold:
                print(f"[ENV] STUCK DETECTED (pause_counter: {self.pause_counter}) - FATAL PENALTY: -20.0")
                terminal_penalty = -20.0
                reward += terminal_penalty
                terminated = True
                self.pause_counter = 0
        else:
            self.pause_counter = 0

        reward_info = {
            "reward_total": float(reward),
            "reward_progress": float(progress_reward),
            "reward_speed": float(speed_reward),
            "reward_side_slip_penalty": float(side_slip_penalty),
            "reward_forward_slip_penalty": float(reverse_penalty),
            "reward_idle_penalty": float(idle_penalty),
            "reward_terminal_bonus": float(terminal_bonus),
            "reward_terminal_penalty": float(terminal_penalty),
            "speed": float(speed),
            "cps_passed": int(cps_passed),
        }

        return state, reward, terminated, False, reward_info


    def _format_state(self, frame, tele):
        tele_vector = np.array([
            tele.get('speed', 0.0),
            tele.get('gear', 0.0),
            tele.get('rpm', 0.0),
            tele.get('slip_forward', 0.0),
            0.0
        ], dtype=np.float32)
        
        if frame is None:
            frame = np.zeros((128, 128, 3), dtype=np.uint8)
            
        return {"vision": frame, "telemetry": tele_vector}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.last_cps = 0
        self.pause_counter = 0
        self.frozen_rpm_counter = 0
        self.last_seen_rpm = -1.0
        
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
                
            print(f"Step: {step:3} | Reward: {reward:6.2f} | CPS: {env.last_cps}")
            
            if terminated:
                print("--- OFF TRACK - RESET ---")
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        cv2.destroyAllWindows()