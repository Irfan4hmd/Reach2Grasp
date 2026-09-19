import robosuite as suite
from robosuite.controllers import load_composite_controller_config
import numpy as np
import argparse
# (check the actual import path/function name in the current robosuite version —
#  it has moved between versions, this is where the docs above matter)

def make_panda_env(render_mode: str = "human"):
    """
    Build and return a robosuite env with:
      - robots="Panda"
      - env_name: pick a simple manipulation task (e.g. "Lift") — read
        robosuite's list of available envs and pick one deliberately,
        note WHY you picked it
      - an explicit OSC_POSE controller config (don't rely on the default)
      - has_renderer / has_offscreen_renderer set according to render_mode
      - control_freq, horizon set to something sane for a smoke test
    Return: the constructed env
    """
    # BASIC controller: arms controlled using OSC, mobile base (if present) using JOINT_VELOCITY, other parts controlled using JOINT_POSITION 
    controller_config = load_composite_controller_config(controller="BASIC")

    # create an environment to visualize on-screen
    env = suite.make(
        "Lift",
        robots=["Panda"],             # load a Panda robot
        gripper_types="default",                # use default grippers per robot arm
        controller_configs=controller_config,   # arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
        # env_configuration="opposed",            # (two-arm envs only) arms face each other
        has_renderer=render_mode == "human",                      # on-screen rendering
        render_camera="frontview",              # visualize the "frontview" camera
        has_offscreen_renderer=render_mode == "headless",           # no off-screen rendering
        control_freq=20,                        # 20 hz control for applied actions
        horizon=200,                            # each episode terminates after 200 steps
        use_object_obs=False,                     # Generates object coordinates/orientations
        use_camera_obs=False,                     # Generates visual RGB-D arrays
        # camera_names=["agentview", "robot0_eye_in_hand"],  # Cameras to capture
        # camera_heights=256,
        # camera_widths=256,
        )
    return env

def run_random_policy(env, n_steps: int = 200, save_frames: bool = False):
    """
    - env.reset()
    - loop n_steps:
        - sample a random action shaped correctly for env.action_dim
          (don't hardcode a shape — read it from the env/action space)
        - env.step(action)
        - if u have a renderer: env.render()
        - if save_frames: grab an offscreen image (via env.sim.render(...)
          or the obs camera dict, depending on how you configured obs) and
          store/save it
    - close env cleanly at the end
    """
    env.reset()
    for i in range(0,n_steps):
        low, high = env.action_spec
        action = np.random.uniform(low, high)
        next_obs, reward, done, info = env.step(action)
        if env.has_renderer:
            env.render()
        # if save_frames == True:
        #     # taking the image from the offscreen camera, which is set up in the env creation
        #     frame = env.sim.render(width=256, height=256, camera_name="agentview")
        #     import imageio
        #     imageio.imwrite(f"frame_{i}.png", frame)
    env.close()
        
    
    
if __name__ == "__main__":
    # parse a --headless flag or similar so you can run this
    # both on-screen (dev machine, display attached) and offscreen
    # (what you'll need for actual training runs later)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no on-screen rendering)")
    parser.add_argument("--save_frames", action="store_true", help="Save frames from the offscreen camera to disk")
    args = parser.parse_args()
    
    render_mode = "headless" if args.headless else "human"
    env = make_panda_env(render_mode=render_mode)
    run_random_policy(env, n_steps=200, save_frames=args.save_frames)
