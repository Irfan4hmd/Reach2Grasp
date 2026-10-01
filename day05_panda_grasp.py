import argparse
import logging
import time
import warnings



warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)   # suppresses robosuite log warnings for now



import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from success_report import SuccessReport


"""
Day 5: Panda grasping demo

This script extends the Day 1 renderer setup into a simple object-interaction demo.
The idea is:
  1. Create a Panda environment with a pick-and-place task.
  2. Reset the environment.
  3. Use a small scripted motion sequence to move toward the object.
  4. Close the gripper, lift the object, and release it.

This is intentionally simple and safe for learning. It is not a full RL policy yet.
"""


def make_panda_pickplace_env(render_mode: str = "human", n_steps: int = 500, seed: int = 0):
    """
    Create a Panda environment suited for a simple grasping task.
    We deliberately use PickPlaceCan because it is a good beginner task:
      - the object is easy to see,
      - the task is goal-driven,
      - it demonstrates manipulation without needing a full custom scene.
    """

    controller_config = load_composite_controller_config(controller="BASIC")
    
    env = suite.make(
        "PickPlaceCan",
        robots=["Panda"],
        gripper_types="default",
        controller_configs=controller_config,
        has_renderer=render_mode == "human",
        has_offscreen_renderer=False,
        render_camera="frontview",
        control_freq=20,
        horizon=n_steps,
        use_object_obs=True,
        use_camera_obs=False,
        seed=seed,
    )
    return env


def get_object_position(env):
    """
    Try a few robust ways to read the object position.
    The exact field names differ a little across robosuite versions.
    """
    if hasattr(env, "objects") and len(env.objects):
        try:
            return np.asarray(env.objects[0].get_position(), dtype=np.float64)
        except Exception:
            pass

    if hasattr(env, "sim") and hasattr(env.sim, "data"):
        try:
            body_names = list(env.sim.model.body_names)
            if body_names:
                body_id = env.sim.model.body_name2id(body_names[-1])
                return np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64)
        except Exception:
            pass

    return np.zeros(3, dtype=np.float64)


def scripted_grasp(env, n_steps: int = 200):
    """
    Run a simple scripted motion pattern for grasping and lifting.

    This is a learning example, not a production controller. It uses a motion
    pattern that you can later replace with IK, controller targets, or a learned
    policy.
    """
    obs = env.reset()
    object_start_z = get_object_position(env)[2]
    object_id = env.object_id
    target_bin = env.target_bin_placements[object_id].copy()
    can = env.objects[object_id]
    can_geom_id = env.obj_geom_id[can.name][0]
    can_half_height = env.sim.model.geom_size[can_geom_id, 2]

    place_pos = target_bin.copy()
    place_pos[2] += can_half_height
    print(f"Place position: {place_pos}")
    phase = "approach"
    close_wait = 0
    stable_steps = 0
    place_stable_steps = 0

    low, high = env.action_spec

    for step in range(n_steps):
        object_pos = get_object_position(env)
        eef_pos = obs.get("robot0_eef_pos")

        if eef_pos is None:
            continue

        action = np.zeros_like(low)

        if phase == "approach":
            # 1) Approach a waypoint above the object
            target_pos = object_pos + np.array([0.0, 0.0, 0.15])
            delta = target_pos - eef_pos

            action[:3] = np.clip(delta[:3], -0.1, 0.1)
            action[3:6] = 0.0
            action[6] = 0.0

            # 2) open the gripper when close to waypoint
            if np.linalg.norm(delta[:3]) < 0.03:
                action[6] = -1.0
                phase = "descend"
                print("Descending toward object...")

        elif phase == "descend":
            # 3) descend toward the object and close the gripper
            delta = object_pos - eef_pos
            action[:3] = np.clip(delta[:3], -0.1, 0.1)
            action[3:6] = 0.0
            action[6] = -1.0

            if np.linalg.norm(delta[:3]) < 0.02:
                print("Closing gripper...")
                action[6] = 1.0  # close gripper
                close_wait+=1
                if close_wait > 5:
                    eef_start_z = eef_pos[2]
                    phase = "lift"
                    
        
        elif phase == "lift":
            # 4) lift the object
            action[:3] = np.array([0.0, 0.0, 0.05])  # lift up
            action[6] = 1.0  # keep gripper closed
            print("Lifting object...")
            if eef_pos[2] - eef_start_z >= 0.05:
                phase = "verify"
            
        # 5) Check object stability after lift
        elif phase == "verify":
            object_lifted = object_pos[2] - object_start_z > 0.05
            if object_lifted:
                stable_steps += 1
            else:
                stable_steps = 0
            if object_lifted and stable_steps >= 5:
                print("grasp success")
                phase = "transport"
        elif phase == "transport":
            # 6) Move to the place position
            
            target_pos = place_pos + np.array([0.0, 0.0, 0.25])  # waypoint above place position
            delta = target_pos - object_pos
            action[:3] = np.clip(delta[:3], -0.1, 0.1)
            if np.linalg.norm(delta[:3]) < 0.05:
                print("Reached place position.")
                phase = "lower"
        elif phase == "lower":
            # 6) Lower the object until it is close to the place position
            
            delta = place_pos - object_pos
            action[:3] = np.clip(delta[:3], -0.05, 0.05)
            print("Lowering object...")
            if np.linalg.norm(delta[:3]) < 0.03:
                print("Reached place position.")
                phase = "release"
        elif phase == "release":
            # 7) Release the object
            action[6] = -1.0  # open gripper
            placed_near_target = np.linalg.norm(place_pos - object_pos) < 0.04
            if placed_near_target:
                place_stable_steps += 1
            else:
                place_stable_steps = 0

            if place_stable_steps >= 5:
                retreat_start_z = eef_pos[2]
                phase = "retreat"
                
        elif phase == "retreat":
            action[:3] = np.array([0.0, 0.0, 0.05])
            action[6] = -1.0
            if eef_pos[2] - retreat_start_z > 0.06:
                task_success = env._check_success()
                print("Robosuite task success:", task_success)
                if task_success:
                    print("Place successful.")
                    return task_success, phase
                

        action = np.clip(action, low, high)
        obs, reward, done, info = env.step(action)
        # if step % 20 == 0 or reward != 0 or done:
        #     print(f"step={step}, reward={reward}, done={done}, info={info}")
        if env.has_renderer:
            env.render()

        if done:
            print(f"Episode ended at step {step}.")
            return env._check_success(), phase
        if env.has_renderer:
            time.sleep(0.05)
    
    return env._check_success(), phase   # return the last phase if we exit the loop without success        

    


def main():
    import warnings
    warnings.filterwarnings("ignore")

    
    parser = argparse.ArgumentParser(description="Day 5: Panda grasping demo")
    parser.add_argument("--headless", action="store_true", help="Run without on-screen rendering")
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be a positive integer")
    render_mode = "headless" if args.headless else "human"
    n_steps = 3000
     # Example place position
    successful_trials = 0
    unsuccessful_grasps_message: list[SuccessReport] = []
    for i in range(args.trials):
        print(f"Running pick-and-place trial {i+1}...")
        env = make_panda_pickplace_env(render_mode=render_mode,n_steps=n_steps)
        task_success, phase = scripted_grasp(env, n_steps=n_steps)
        if task_success:
            successful_trials += 1
        else:
            report = SuccessReport()
            report.successful = False
            report.last_phase = phase
            report.trial_number = i + 1
            report.object_to_target_distance = np.linalg.norm(get_object_position(env) - env.target_bin_placements[env.object_id])
            unsuccessful_grasps_message.append(report)
        env.close()
        
    print(f"Successful pick-and-place trials: {successful_trials} out of {args.trials}")
    for message in unsuccessful_grasps_message:
        print(f"Pick-and-place trial {message.trial_number} failed with can-to-target distance: {message.object_to_target_distance:.3f}; last phase: {message.last_phase}")
    
    success_rate = successful_trials / args.trials
    print(f"Overall success rate: {success_rate:.2%}")
    
if __name__ == "__main__":
    main()
