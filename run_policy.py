import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import os
from saved_goto_point import GoToPointWrapper

# --- Create and wrap environment ---
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
    render_mode="human",  # change to "human" to visualize
)

env = GoToPointWrapper(env)

# --- Load the model ---
model_path = "ppo_ant_goto.zip"
vec_env = make_vec_env(lambda: env, n_envs=1)
print(f"🔁 Loading existing model from {model_path}...")
model = PPO.load(model_path, env=vec_env)

# --- Test the trained agent ---
test_env = GoToPointWrapper(env)
obs, info = test_env.reset()

# Access the internal simulation environment (MuJoCo)
sim_env = test_env.unwrapped

# --- Set custom start state ---
sim_env.data.qpos[:] = np.array([10.0, 2.0, 0.0, 0.5, 0.0, 0.0, 0.0, 11.0, 1.0])  # Set the joint positions
# sim_env.data.qvel[:] = np.zeros_like(custom_qpos)  # Set the joint velocities
# ----------------
for _ in range(50000):
    print(sim_env.data.qpos)



# print("🎯 Testing trained agent...")
# # Run the simulation and test the agent
# for _ in range(500):
#     # current_pos = np.copy(sim_env.data.qpos)
#     # print(current_pos)
#     # print("---")

#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = test_env.step(action)
#     test_env.render()

#     if terminated or truncated:
#         obs, info = test_env.reset()

# test_env.close()
print("✅ Done!")
