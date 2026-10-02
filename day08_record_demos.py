import argparse
import os
import time

import numpy as np

from custom_exceptions.phase_timeout import PhaseTimeout
from day05_panda_grasp import make_panda_pickplace_env
from day07_panda_reach_and_grasp import ScriptedPickPlacePolicy


OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "Can_pos", "Can_quat"]  # one constant, fixed order
PATH = "demos/panda_pickplace_demos.npz"
class DemoRecorder:
    def __init__(self): 
        self.obs_buffer = []
        self.action_buffer = []
        self.episodes = []
        self.current_seed = 0
    def start(self, env, seed, n_steps=500):
        # reset per-episode buffers
        obs = env.reset()
        policy = ScriptedPickPlacePolicy()
        policy.reset(env, obs)
        
        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.current_seed = seed
        self.run_and_record(env, obs, policy, n_steps)

    def run_and_record(self, env, obs, policy, n_steps=500):
        successful = False
        reason = "horizon"
        steps = 0
        for step in range(n_steps):
            try:
                action = policy.act(obs=obs)
                self.record(obs, action)
                obs, reward, done, info = env.step(action)
                steps += 1
                if env.has_renderer:
                    env.render()
                    time.sleep(0.05)  # Add a small delay for visualization
                
                if policy.phase == "retreat" and env._check_success():
                    successful = True
                    reason = "success"
                    break
                if done:
                    break
            except PhaseTimeout as e:
                reason = str(e)
                print(f"Episode terminated due to: {reason}")
                break
            
        self.end(successful)
    
    
    def record(self, obs, action): 
        # recording the observation, action, and seed for the current step
        # flatten obs via OBS_KEYS, append
        flattened_obs = np.concatenate([obs[key] for key in OBS_KEYS])
        self.obs_buffer.append(flattened_obs)
        self.action_buffer.append(action)
        
    def end(self, success): 
        # keep the episode only if success
        if success:
            obs_buffer = np.array(self.obs_buffer)
            action_buffer = np.array(self.action_buffer)
            self.episodes.append((obs_buffer, action_buffer, self.current_seed))
        self.obs_buffer.clear()
        self.action_buffer.clear()
    def save(self, path): 
        
        # save all episodes to a file, along with the success reports
        demo_dict = {}
        for index, (obs_buffer, action_buffer, seed) in enumerate(self.episodes):
            demo_dict[f"demo{index}_obs"] = obs_buffer
            demo_dict[f"demo{index}_actions"] = action_buffer
            demo_dict[f"demo{index}_seed"] = seed
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **demo_dict)
        


def replay(path,render_mode="headless"):
    # replay the demos from the file
    with np.load(path) as data:
        
        demo_action_keys = [key for key in data.files if key.startswith("demo") and key.endswith("_actions")]
        success_count = 0
        for action_key in demo_action_keys:
            k = action_key.split("_")[0][4:]  # extract the demo index
            seed = int(data[f"demo{k}_seed"])
            env = make_panda_pickplace_env(render_mode=render_mode, n_steps=3000, seed=seed)
            obs = env.reset()
            for action in data[action_key]:
                obs, reward, done, info = env.step(action)
                if env.has_renderer:
                    env.render()
                    time.sleep(0.05)  # Add a small delay for visualization
                if done:
                    break
            success = env._check_success()
            if success:
                success_count += 1
            env.close()
            print(f"demo {k}: replay. Success: {success}")
        print(f"Total successful replays: {success_count} out of {len(demo_action_keys)}")
        
            
            
def main():
    import warnings
    warnings.filterwarnings("ignore")

    
    parser = argparse.ArgumentParser(description="Day 8: Panda grasping demo recorder")
    parser.add_argument("--headless", action="store_true", help="Run without on-screen rendering")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--replay", action="store_true", help="Replay the recorded demos")
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be a positive integer")
    render_mode = "headless" if args.headless else "human"
    n_steps = 3000
     # recording the demos
    if args.replay:
        replay(PATH,render_mode=render_mode)
        
    else:
        recorder = DemoRecorder()
        for i in range(args.trials):
            print(f"Running pick-and-place trial {i+1}...")
            env = make_panda_pickplace_env(render_mode=render_mode,n_steps=n_steps,seed=args.seed+i)
            recorder.start(env, seed=args.seed+i, n_steps=n_steps)
            env.close()
        recorder.save(PATH)
        with np.load(PATH) as data:
            for key in data.files:
                print(f"{key}: shape {data[key].shape}")

if __name__ == "__main__":
    main()