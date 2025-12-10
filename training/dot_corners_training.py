# Import necessary libraries
import pygame
import sys
import numpy as np # For the Q-table and calculations
import math # For distance calculations
import random
import policy_drawing_correspondences
from os.path import join

# --- Constants ---
# Screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

# Colors (R, G, B)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0) # For the target square
RED = (255, 0, 0) # For the dot (to see it better)

# Dot properties
DOT_RADIUS = 8
DOT_SPEED = 5 # This is now the agent's action magnitude

SAVE_DIR = "./trained_policies/"
# --- Target Square Definition ---
# Define the corners of the square path
# --- Top Left ---
TOP_LEFT_X = 14
TOP_LEFT_Y = 12
FILE_NAME = "q_table_topleft.npy"
# --- Top Right ---
# TOP_LEFT_X = 559
# TOP_LEFT_Y = 12
# FILE_NAME = "q_table_topright.npy"
# --- Bottom Right ---
# TOP_LEFT_X = 559
# TOP_LEFT_Y = 559
# FILE_NAME = "q_table_bottomright.npy"
# --- Bottom Left ---
# TOP_LEFT_X = 12
# TOP_LEFT_Y = 559
# FILE_NAME = "q_table_bottomleft.npy"

# save the info for visualizing
policy_drawing_correspondences.save_policy_info(FILE_NAME, {"type":"point", "x":TOP_LEFT_X, "y":TOP_LEFT_Y, "r":None})

Q_TABLE_FILE = join(SAVE_DIR, FILE_NAME)

# --- Q-Learning Parameters ---
# Discretize the state space
GRID_SIZE = 20 # 20x20 pixel cells
STATES_X = SCREEN_WIDTH // GRID_SIZE   # 30 states
STATES_Y = SCREEN_HEIGHT // GRID_SIZE  # 30 states

# --- NEW: Mode and File Constants ---
MODE = "run"  # Change to "run" to execute the learned policy
# Q_TABLE_FILE = "q_table_topright.npy"


# --- Helper Functions ---

def get_state(x, y):
    """Converts (x, y) coordinates to a discrete grid state."""
    state_x = int(max(0, min(x, SCREEN_WIDTH - 1)) // GRID_SIZE)
    state_y = int(max(0, min(y, SCREEN_HEIGHT - 1)) // GRID_SIZE)
    return (state_x, state_y)

def get_reward(x, y):
    """Calculates the reward based on distance to the point."""
    p = (x, y)
    # Find distance to the closest of the 4 line segments
    return -1 * np.linalg.norm(p-np.array((TOP_LEFT_X,TOP_LEFT_Y)))

    # Reward is the negative distance.
    # We want to *minimize* distance, which means *maximizing* this reward.


def choose_action(state, epsilon, q_table):
    """Epsilon-greedy action selection."""
    if np.random.rand() < epsilon:
        return np.random.randint(4) # Explore: random action
    else:
        # Exploit: choose best action from Q-table
        return np.argmax(q_table[state[0], state[1], :])

def main():
    """Main function to run training or execution."""
    # --- Initialization ---
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Q-Learning Dot - {MODE.upper()} MODE")
    clock = pygame.time.Clock()

    if MODE == "train":
        print("--- Mode: TRAINING ---")
        
        # --- Q-Table and Learning Params ---
        # Q-Table: (state_x, state_y, action)
        # 4 actions: 0=Up, 1=Down, 2=Left, 3=Right
        q_table = np.zeros((STATES_X, STATES_Y, 4))

        # Learning parameters
        ALPHA = 0.1      # Learning rate
        GAMMA = 0.9      # Discount factor
        EPSILON = 1.0    # Exploration rate
        EPSILON_DECAY = 0.999
        MIN_EPSILON = 0.01

        # Training parameters
        NUM_EPISODES = 80000
        STEPS_PER_EPISODE = 100

        # --- Main Training Loop ---
        for episode in range(NUM_EPISODES):
            # Reset dot to the center for each new episode
            # dot_x = SCREEN_WIDTH // 2
            # dot_y = SCREEN_HEIGHT // 2
            # reset dot to random location for each new episode
            dot_x = random.randint(0, SCREEN_WIDTH)
            dot_y = random.randint(0, SCREEN_HEIGHT)
            
            for step in range(STEPS_PER_EPISODE):
                # --- Event Handling (to keep window responsive) ---
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        # If quitting mid-training, save progress
                        print("Training interrupted. Saving Q-table...")
                        np.save(Q_TABLE_FILE, q_table)
                        print(f"Q-table saved to {Q_TABLE_FILE}")
                        pygame.quit()
                        sys.exit()

                # --- Q-Learning Agent Logic ---
                # 1. Get current state
                current_state = get_state(dot_x, dot_y)
                
                # 2. Choose action
                action = choose_action(current_state, EPSILON, q_table)
                
                # 3. Take action (move the dot)
                if action == 0: # Up
                    dot_y -= DOT_SPEED
                elif action == 1: # Down
                    dot_y += DOT_SPEED
                elif action == 2: # Left
                    dot_x -= DOT_SPEED
                elif action == 3: # Right
                    dot_x += DOT_SPEED

                # 4. Boundary check
                dot_x = max(DOT_RADIUS, min(SCREEN_WIDTH - DOT_RADIUS, dot_x))
                dot_y = max(DOT_RADIUS, min(SCREEN_HEIGHT - DOT_RADIUS, dot_y))

                # 5. Observe new state and reward
                new_state = get_state(dot_x, dot_y)
                reward = get_reward(dot_x, dot_y)
                
                # 6. Update Q-Table (Bellman equation)
                old_q = q_table[current_state[0], current_state[1], action]
                max_future_q = np.max(q_table[new_state[0], new_state[1], :])
                
                new_q = old_q + ALPHA * (reward + GAMMA * max_future_q - old_q)
                q_table[current_state[0], current_state[1], action] = new_q

                # --- Drawing ---
                # Only draw every N episodes to speed up training
                if episode % 100 == 0:
                    screen.fill(BLACK)
                    
                    # Draw the target square
                    pygame.draw.circle(screen, GREEN, (TOP_LEFT_X, TOP_LEFT_Y), DOT_RADIUS)
                    
                    # Draw the agent's dot
                    pygame.draw.circle(screen, RED, (int(dot_x), int(dot_y)), DOT_RADIUS)

                    # --- Update Display ---
                    pygame.display.flip()
                    clock.tick(60) # Control the speed of visualization

            # --- End of Episode ---
            # Decay epsilon
            EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
            
            if episode % 100 == 0:
                print(f"Episode {episode} finished. Epsilon: {EPSILON:.4f}")

        # --- End of Training ---
        print("Training finished.")
        np.save(Q_TABLE_FILE, q_table)
        print(f"Q-table saved to {Q_TABLE_FILE}")

    elif MODE == "run":
        print("--- Mode: RUNNING POLICY ---")
        
        # --- Load Q-Table ---
        try:
            q_table = np.load(Q_TABLE_FILE)
            print(f"Loaded Q-table from {Q_TABLE_FILE}")
        except FileNotFoundError:
            print(f"Error: Q-table file '{Q_TABLE_FILE}' not found.")
            print("Please run the script in 'train' mode first.")
            pygame.quit()
            sys.exit()

        # --- Run Policy Loop ---
        dot_x = SCREEN_WIDTH // 2
        dot_y = SCREEN_HEIGHT // 2
        
        running = True
        while running:
            # --- Event Handling (to allow quitting) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # --- Agent Logic (Exploitation only) ---
            # 1. Get current state
            current_state = get_state(dot_x, dot_y)
            
            # 2. Choose action (epsilon=0 for pure exploitation)
            action = choose_action(current_state, 0, q_table)
            
            # 3. Take action (move the dot)
            if action == 0: # Up
                dot_y -= DOT_SPEED
            elif action == 1: # Down
                dot_y += DOT_SPEED
            elif action == 2: # Left
                dot_x -= DOT_SPEED
            elif action == 3: # Right
                dot_x += DOT_SPEED

            # 4. Boundary check
            dot_x = max(DOT_RADIUS, min(SCREEN_WIDTH - DOT_RADIUS, dot_x))
            dot_y = max(DOT_RADIUS, min(SCREEN_HEIGHT - DOT_RADIUS, dot_y))

            # 5. NO reward or Q-table update in 'run' mode

            # --- Drawing (every frame) ---
            screen.fill(BLACK)
            pygame.draw.circle(screen, GREEN, (TOP_LEFT_X, TOP_LEFT_Y), DOT_RADIUS)
            pygame.draw.circle(screen, RED, (int(dot_x), int(dot_y)), DOT_RADIUS)
            
            # --- Update Display ---
            pygame.display.flip()
            clock.tick(60) # Run at a viewable speed
    
    else:
        print(f"Error: Invalid MODE '{MODE}'. Please set to 'train' or 'run'.")

    # --- Shutdown ---
    pygame.quit()
    sys.exit()

# --- Main Guard ---
if __name__ == "__main__":
    main()