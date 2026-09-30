import argparse
import logging
import time
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)   # suppresses robosuite log warnings for now



import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config


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


def make_panda_pickplace_env(render_mode: str = "human", n_steps: int = 500):
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

    grasped = False
    lifted = False
    close_wait = 0
    lift_height = 0.0

    low, high = env.action_spec

    for step in range(n_steps):
        object_pos = get_object_position(env)
        eef_pos = obs.get("robot0_eef_pos")

        if eef_pos is None:
            continue

        action = np.zeros_like(low)

        if not grasped:
            # 1) Approach a waypoint above the object
            target_pos = object_pos + np.array([0.0, 0.0, 0.15])
            delta = target_pos - eef_pos

            action[:3] = np.clip(delta[:3], -0.05, 0.05)
            action[3:6] = 0.0
            action[6] = 0.0

            # 2) Close gripper when close enough
            if np.linalg.norm(delta[:3]) < 0.03:
                action[6] = -1.0
                close_wait += 1
                if close_wait >= 5:
                    grasped = True
                    print("Object grasped, waiting to lift...")

        elif not lifted:
            # 3) Wait after close
            action[:3] = np.array([0.0, 0.0, 0.0])
            action[3:6] = 0.0
            action[6] = -1.0

            if close_wait >= 5:
                # 4) Lift upward
                action[:3] = np.array([0.0, 0.0, 0.04])
                action[6] = -1.0
                lift_height += 0.04

                if lift_height > 0.12:
                    lifted = True

        # 5) Check object stability after lift
        if lifted:
            object_dist = np.linalg.norm(object_pos - eef_pos)
            print("distance to object after lift:", object_dist)

            if object_dist < 0.05:
                print("grasp success")
                break

        action = np.clip(action, low, high)
        obs, reward, done, info = env.step(action)

        if env.has_renderer:
            env.render()

        if done:
            print(f"Episode ended at step {step}.")
            break

        time.sleep(0.05)

    env.close()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    
    parser = argparse.ArgumentParser(description="Day 5: Panda grasping demo")
    parser.add_argument("--headless", action="store_true", help="Run without on-screen rendering")
    args = parser.parse_args()
    
    render_mode = "headless" if args.headless else "human"
    n_steps = 500
    env = make_panda_pickplace_env(render_mode=render_mode,n_steps=n_steps)
    scripted_grasp(env, n_steps=n_steps)


if __name__ == "__main__":
    main()
