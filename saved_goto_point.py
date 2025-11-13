import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import os

# --- Custom Action Wrapper to freeze Z motion ---
class XYOnlyActionWrapper(gym.ActionWrapper):
    def __init__(self, env, z_index=2):
        super().__init__(env)
        self.z_index = z_index
        self.static_z_value = None  # to keep z constant

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Record the current z position (e.g., end-effector height)
        self.static_z_value = float(self.env.unwrapped.data.qpos[self.z_index])
        return obs, info

    def action(self, action):
        # Clone the action array to modify safely
        new_action = np.copy(action)
        # Freeze Z movement (set to 0 if it’s velocity control, or static if position control)
        new_action[self.z_index] = 0.0
        return new_action




# --- Custom Wrapper for goal-reaching task ---
class GoToPointWrapper(gym.Wrapper):
    def __init__(self, env, target=np.array([2.0, 2.0])):
        super().__init__(env)
        self.target = np.array(target, dtype=np.float32)
        self.sim_env = env.unwrapped  # get underlying mujoco env

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.start_pos = np.copy(self.sim_env.data.qpos[:2])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_pos = np.copy(self.sim_env.data.qpos[:2])

        dist_to_target = np.linalg.norm(current_pos - self.target)
        reward = -dist_to_target

        # Bonus for reaching target
        if dist_to_target < 0.3:
            reward += 10.0
            terminated = True

        return obs, reward, terminated, truncated, info


def main():

# if __name__ == "__main__":
#     main()
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
    # --------- Restricts actions to x-y -----------
    # env = XYOnlyActionWrapper(env)
    env = GoToPointWrapper(env)

    # --- RL training setup ---
    model_path = "ppo_reacher_goto.zip"
    vec_env = make_vec_env(lambda: env, n_envs=1)

    if os.path.exists(model_path):
        print(f"🔁 Loading existing model from {model_path}...")
        model = PPO.load(model_path, env=vec_env)
    else:
        print("🚀 Training new model...")
        model = PPO("MlpPolicy", vec_env, verbose=1)
        model.learn(total_timesteps=200_000)
        model.save(model_path)
        print(f"✅ Model saved to {model_path}")

    # --- Test the trained agent ---
    test_env = GoToPointWrapper(env)
    obs, info = test_env.reset()

    print("🎯 Testing trained agent...")
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        test_env.render()
        if terminated or truncated:
            obs, info = test_env.reset()

    test_env.close()
    print("✅ Done!")
