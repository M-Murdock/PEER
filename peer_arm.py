import mujoco
import gymnasium as gym
import numpy as np
import os
# # Loading a specific model description as an imported module.
# from robot_descriptions import panda_mj_description
# model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)

# # Directly loading an instance of MjModel.
# from robot_descriptions.loaders.mujoco import load_robot_description
# model = load_robot_description("panda_mj_description")

# # Loading a variant of the model, e.g. panda without a gripper.
# # model = load_robot_description("panda_mj_description", variant="panda_nohand")

# frame = model.render()


env = gym.make(
    "Ant-v5",
    xml_file="./mujoco_menagerie/franka_emika_panda/scene.xml",
    forward_reward_weight=1,
    ctrl_cost_weight=0.05,
    contact_cost_weight=5e-4,
    healthy_reward=1,
    main_body=1,
    healthy_z_range=(0.195, 0.75),
    include_cfrc_ext_in_observation=True,
    exclude_current_positions_from_observation=False,
    reset_noise_scale=0.1,
    frame_skip=25,
    max_episode_steps=1000,
    render_mode="human",  # Change to "human" to visualize
)

# Example of running the environment for a few steps
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # Replace with your agent's action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("Environment tested successfully!")