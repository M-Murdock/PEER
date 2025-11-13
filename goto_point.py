import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

# --- Custom Wrapper for goal-reaching task ---
class GoToPointWrapper(gym.Wrapper):
    def __init__(self, env, target=np.array([2.0, 2.0])):
        super().__init__(env)
        self.target = np.array(target, dtype=np.float32)

        # Get a handle to the underlying mujoco env (unwrapped)
        self.sim_env = env.unwrapped  # now we can safely use .data.qpos etc.

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # store starting position
        self.start_pos = np.copy(self.sim_env.data.qpos[:2])  
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Access mujoco state safely
        current_pos = np.copy(self.sim_env.data.qpos[:2])

        # Compute distance-based reward
        dist_to_target = np.linalg.norm(current_pos - self.target)
        reward = -dist_to_target  # negative distance

        # Give bonus for reaching target
        if dist_to_target < 0.3:
            reward += 10.0
            terminated = True

        return obs, reward, terminated, truncated, info


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
    render_mode="human",  # set to "human" to visualize
)

env = GoToPointWrapper(env)

# --- Train RL agent ---
vec_env = make_vec_env(lambda: env, n_envs=1)
model = PPO("MlpPolicy", vec_env, verbose=1)

model.learn(total_timesteps=200_000)

# --- Test trained policy ---
test_env = GoToPointWrapper(env)
obs, info = test_env.reset()

for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    test_env.render()
    if terminated or truncated:
        obs, info = test_env.reset()

test_env.close()
print(" complete and agent tested.")
