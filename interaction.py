import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

class BlackBox():
    def __init__(self, start_pos=[0.5, 0.5]):
        # --- Configuration ---
        self.ENV_ID = "Reacher-v5"  # Updated to use the modern Reacher-v5 MuJoCo environment
        self.TOTAL_TIMESTEPS = 100 # Number of steps to train the agent
        self.LOG_DIR = "./reacher_ppo_logs_v5/" # Updated log directory
        self.MODEL_PATH = "reacher_ppo_model_v5.zip" # Updated model path

        # Custom start position for the two arm joints (qpos[0] and qpos[1]) in radians.
        # Example: [0.0, 0.0] is straight out. [0.5, 0.5] is slightly bent.
        # The Reacher-v5 state space is two joint angles.
        self.start_pos = start_pos

    def run(self):
        """
        Loads the saved model and runs a quick evaluation episode, 
        optionally setting a custom start joint position.
        """
        if not os.path.exists(self.MODEL_PATH):
            print(f"Error: Model file not found at {self.MODEL_PATH}. Please run training first.")
            return

        print(f"\nEvaluating saved model: {self.MODEL_PATH}")
        if self.start_pos is not None:
            print(f"Using custom start joint positions (qpos[:2]): {self.start_pos}")

        # Load the trained agent
        loaded_model = PPO.load(self.MODEL_PATH)

        # Create a single environment for rendering/testing
        # We use render_mode="human" for visualization
        eval_env = gym.make(self.ENV_ID, render_mode="human")

        # 1. Reset the environment (initializes the internal MuJoCo state)
        obs, info = eval_env.reset()
         
        if self.start_pos is not None:
            print("NOT NONE")
            # MuJoCo environments expose the raw environment via .unwrapped
            # This allows direct manipulation of the simulation data.
            sim_data = eval_env.unwrapped.data
            
            # The Reacher-v5 qpos array still has 4 elements: [joint0, joint1, target_x, target_y]
            # We only set the first two elements (the arm joints).
            try:
                # Set joint positions
                sim_data.qpos[:len(self.start_pos)] = self.start_pos
                
                # Reset joint velocities (optional, but good practice)
                # sim_data.qvel[:] = 0.0

                # Forward kinematics: Must be called after state manipulation to update
                # all derived quantities (like link positions and contact forces).
                eval_env.unwrapped.sim.forward()
                
                # Get the new observation after setting the state
                # This relies on the internal MujocoEnv method _get_obs
                obs = eval_env.unwrapped._get_obs()
                
            except Exception as e:
                print(f"Error setting custom start state: {e}")
                print("Falling back to default reset state.")
                # If state setting fails, just use the obs from the initial reset()

        episode_reward = 0
        terminated = False
        truncated = False

        # Run one episode
        while not terminated and not truncated:
            # Predict the action using the loaded model
            action, _ = loaded_model.predict(obs, deterministic=True)

            # Take the action in the environment
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward

        eval_env.close()
        print(f"Evaluation finished. Total reward for the episode: {episode_reward:.2f}")
        
        
    def train(self):
        """
        Sets up the environment, trains a PPO agent, and saves the model.
        Training is still done with randomized starting positions for robustness.
        """
        print(f"Starting training for {self.ENV_ID}...")

        # Create environment
        env = gym.make("Reacher-v5")   # or Reacher-v5

        # Create PPO model
        model = PPO("MlpPolicy", env, verbose=1)

        # Train
        model.learn(total_timesteps=self.TOTAL_TIMESTEPS)

        # ----------------------------------------------------
        # SAVE THE TRAINED MODEL
        # ----------------------------------------------------
        model.save(self.MODEL_PATH)      # saves ppo_reacher_model.zip
        print("Model saved to ppo_reacher_model.zip")

        # Optional: close env after training
        env.close()
        print(f"\nTraining finished. Model saved to {self.MODEL_PATH}")    
    
    
if __name__ == "__main__":
   black_box = BlackBox(start_pos=np.array([-3.14, -1]) )
   black_box.train()
   black_box.run()
   