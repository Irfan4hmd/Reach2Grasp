
import argparse
import time

import numpy as np
from custom_exceptions.phase_timeout import PhaseTimeout
from day05_panda_grasp import make_panda_pickplace_env
from success_report import SuccessReport


class ScriptedPickPlacePolicy:
    


    def reset(self,env,obs):
        self.phase = "approach"
        self.close_wait = 0
        self.stable_steps = 0
        self.eef_start_z = None
        
        self.object_start_z = obs["Can_pos"][2]
        self.can_half_height = env.sim.model.geom_size[env.obj_geom_id[env.objects[env.object_id].name][0], 2]
        self.place_pos = env.target_bin_placements[env.object_id].copy()
        self.place_pos[2] += self.can_half_height
        self.low, self.high = env.action_spec
        self.place_stable_steps = 0
        self.phase_steps = {
            "approach": 0,
            "descend": 0,
            "lift": 0,
            "verify": 0,
            "transport": 0,
            "lower": 0,
            "release": 0,
            "retreat": 0
        }
        self.phase_budgets = {'approach': 840, 'descend': 537, 'lift': 252, 'verify': 15, 'transport': 1914, 'lower': 876, 'release': 15, 'retreat': 51}
        
        
        
    
    def act(self,obs):
        
        object_pos = obs["Can_pos"]
        eef_pos = obs["robot0_eef_pos"]
        place_pos = self.place_pos
        low, high = self.low, self.high

        action = np.zeros_like(low)
        self.phase_steps[self.phase] += 1
        if self.phase_steps[self.phase] > self.phase_budgets[self.phase]:
            raise PhaseTimeout(f"{self.phase}")

        if self.phase == "approach":
            # 1) Approach a waypoint above the object
            target_pos = object_pos + np.array([0.0, 0.0, 0.15])
            delta = target_pos - eef_pos

            action[:3] = np.clip(delta[:3], -0.1, 0.1)
            action[3:6] = 0.0
            action[6] = 0.0

            # 2) open the gripper when close to waypoint
            if np.linalg.norm(delta[:3]) < 0.03:
                action[6] = -1.0
                self.phase = "descend"
                print("Descending toward object...")

        elif self.phase == "descend":
            delta = object_pos - eef_pos
            action[:3] = np.clip(delta[:3], -0.1, 0.1)
            action[3:6] = 0.0
            action[6] = -1.0

            if np.linalg.norm(delta[:3]) < 0.02:
                print("Closing gripper...")
                action[6] = 1.0  # close gripper
                self.close_wait+=1
                if self.close_wait > 5:
                    self.eef_start_z = eef_pos[2]
                    self.phase = "lift"
                    print("Lifting object...")
                    
        
        elif self.phase == "lift":
            # 4) lift the object
            action[:3] = np.array([0.0, 0.0, 0.05])  # lift up
            action[6] = 1.0  # keep gripper closed
            # Initialize eef_start_z on the first step
            if eef_pos[2] - self.eef_start_z >= 0.05:
                self.phase = "verify"
            
        # 5) Check object stability after lift
        elif self.phase == "verify":
            object_lifted = object_pos[2] - self.object_start_z > 0.05
            if object_lifted:
                self.stable_steps += 1
            else:
                self.stable_steps = 0
            if object_lifted and self.stable_steps >= 5:
                print("grasp success")
                self.phase = "transport"
        elif self.phase == "transport":
            # 6) Move to the place position
            target_pos = place_pos + np.array([0.0, 0.0, 0.25])  # waypoint above place position
            delta = target_pos - object_pos
            action[:3] = np.clip(delta[:3], -0.1, 0.1)
            if np.linalg.norm(delta[:3]) < 0.05:
                print("Reached place position.")
                self.phase = "lower"
                print("Lowering object...")
        elif self.phase == "lower":
            # 6) Lower the object until it is close to the place position
            delta = place_pos - object_pos
            action[:3] = np.clip(delta[:3], -0.05, 0.05)
            if np.linalg.norm(delta[:3]) < 0.03:
                print("Reached place position.")
                self.phase = "release"
                
        elif self.phase == "release":
            # 7) Release the object
            action[6] = -1.0  # open gripper
            placed_near_target = np.linalg.norm(place_pos - object_pos) < 0.04
            if placed_near_target:
                self.place_stable_steps += 1
            else:
                self.place_stable_steps = 0

            if self.place_stable_steps >= 5:
                self.retreat_start_z = eef_pos[2]
                self.phase = "retreat"
                
        elif self.phase == "retreat":
            action[:3] = np.array([0.0, 0.0, 0.05])
            action[6] = -1.0
            
        action = np.clip(action, self.low, self.high)
        
        return action

    def run_episode(self, env, n_steps: int = 200):
        obs = env.reset()
        self.reset(env,obs)
        successful = False
        last_phase = None
        steps = 0
        reason = "horizon"
        for step in range(n_steps):
            try:
                action = self.act(obs=obs)
                obs, reward, done, info = env.step(action)
                steps += 1
                last_phase = self.phase
                if env.has_renderer:
                    env.render()
                    time.sleep(0.05)  # Add a small delay for visualization
                
                
                if self.phase == "retreat" and env._check_success():
                    successful = True
                    reason = "success"
                    break
                if done:
                    break
            except Exception as e:
                reason = str(e)
                print(f"Episode terminated due to: {reason}")
                break
            
        return successful,reason, last_phase, steps ,obs

    
def main():
    import warnings
    warnings.filterwarnings("ignore")

    
    parser = argparse.ArgumentParser(description="Day 7: Panda grasping demo")
    parser.add_argument("--headless", action="store_true", help="Run without on-screen rendering")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--calibrate", action="store_true", help="Run calibration trials to determine phase budgets")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be a positive integer")
    render_mode = "headless" if args.headless else "human"
    n_steps = 3000
     # Example place position
    successful_trials = 0
    successreports: list[SuccessReport] = []
    for i in range(args.trials):
        print(f"Running pick-and-place trial {i+1}...")
        env = make_panda_pickplace_env(render_mode=render_mode,n_steps=n_steps,seed=args.seed+i)
        policy = ScriptedPickPlacePolicy()
        successful,reason, last_phase, steps ,obs = policy.run_episode(env, n_steps=n_steps)
        report = SuccessReport()
        report.last_phase = last_phase
        report.trial_number = i + 1
        report.object_to_target_distance = np.linalg.norm(obs["Can_pos"] - policy.place_pos)
        report.steps = steps
        report.phase_steps = policy.phase_steps
        report.reason = reason
        if successful:
            successful_trials += 1
            report.successful = True
        else:
            report.successful = False
        
        successreports.append(report)
        env.close()
        
    print(f"Successful pick-and-place trials: {successful_trials} out of {args.trials}")
    
    
    success_rate = successful_trials / args.trials
    # this block is to calculate the phase budgets 
    if args.calibrate:
        from utils.set_phase_budgets import set_phase_budgets
        phase_budgets = set_phase_budgets(successreports)
        print(f"Phase budgets: {phase_budgets}")
    # mean and std of steps for successful trials
    successful_steps = [report.steps for report in successreports if report.successful]
    if successful_steps:
        mean_steps = np.mean(successful_steps)
        std_steps = np.std(successful_steps)
        print(f"Mean steps for successful trials: {mean_steps:.2f}, Std: {std_steps:.2f}")
    else:
        print("No successful trials to calculate mean and std of steps.")
        
    # failure histogram by reason
    failure_reasons = [report.reason for report in successreports if not report.successful]
    if failure_reasons:
        unique_reasons, counts = np.unique(failure_reasons, return_counts=True)
        print("Failure reasons and counts:")
        for reason, count in zip(unique_reasons, counts):
            print(f"{reason}: {count}")
    
        
    print(f"Overall success rate: {success_rate:.2%}")

if __name__ == "__main__":
    main()
    
