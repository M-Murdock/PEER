import pygame
import sys
import numpy as np
import math
import random

# --- Constants ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

DOT_RADIUS = 8
DOT_SPEED = 5 

# --- Circle Constants ---
CIRCLE_CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
CIRCLE_RADIUS = 200 

Q_TABLE_FILE = "q_table_orbit.npy"

# --- Q-Learning Parameters ---
GRID_SIZE = 20 
STATES_X = SCREEN_WIDTH // GRID_SIZE 
STATES_Y = SCREEN_HEIGHT // GRID_SIZE 

MODE = "run" 

def get_state(x, y):
    state_x = int(max(0, min(x, SCREEN_WIDTH - 1)) // GRID_SIZE)
    state_y = int(max(0, min(y, SCREEN_HEIGHT - 1)) // GRID_SIZE)
    return (state_x, state_y)

def get_reward(current_x, current_y, dx, dy):
    """
    Combines two rewards:
    1. POSITION: Penalty for being away from the ring radius.
    2. MOTION: Reward for moving Counter-Clockwise (Tangent).
    """
    # --- 1. Position Component (Gravity) ---
    # Vector from center to agent
    rel_x = current_x - CIRCLE_CENTER[0]
    rel_y = current_y - CIRCLE_CENTER[1]
    
    dist_from_center = math.sqrt(rel_x**2 + rel_y**2)
    
    # Penalty is distance from the ideal radius
    dist_error = abs(dist_from_center - CIRCLE_RADIUS)
    
    # We use a sharp penalty so it REALLY wants to stay on the line
    reward_position = -dist_error 

    # --- 2. Motion Component (Spin) ---
    # Calculate the ideal Tangent Vector for Counter-Clockwise motion
    # For screen coords (y is down): Tangent is (rel_y, -rel_x)
    tangent_x = rel_y
    tangent_y = -rel_x
    
    # Normalize Tangent Vector (so magnitude doesn't affect reward, only direction)
    if dist_from_center == 0: 
        tangent_len = 1 # Avoid div by zero
    else:
        tangent_len = dist_from_center
        
    norm_tan_x = tangent_x / tangent_len
    norm_tan_y = tangent_y / tangent_len
    
    # Dot Product: (Move Vector) dot (Tangent Vector)
    # This measures how much of our move is "aligned" with the tangent
    alignment = (dx * norm_tan_x) + (dy * norm_tan_y)
    
    # Scale the spin reward (Strong enough to motivate, weak enough not to break gravity)
    reward_spin = alignment * 50 

    # --- Total Reward ---
    return reward_position + reward_spin

def choose_action(state, epsilon, q_table):
    if np.random.rand() < epsilon:
        return np.random.randint(4) 
    else:
        return np.argmax(q_table[state[0], state[1], :])

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Q-Learning Orbit - {MODE.upper()}")
    clock = pygame.time.Clock()

    q_table = np.zeros((STATES_X, STATES_Y, 4))
    
    # Training Params
    ALPHA = 0.1      
    GAMMA = 0.9      
    EPSILON = 1.0    
    EPSILON_DECAY = 0.9996 # Slow decay for complex movement
    MIN_EPSILON = 0.01
    NUM_EPISODES = 80000   # Needs time to learn the flow

    if MODE == "run":
        try:
            q_table = np.load(Q_TABLE_FILE)
            EPSILON = 0
        except:
            print("Train first!")
            sys.exit()

    for episode in range(NUM_EPISODES):
        # dot_x, dot_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        dot_x = random.randint(0, SCREEN_WIDTH)
        dot_y = random.randint(0, SCREEN_HEIGHT)
        
        # Run longer episodes so it can complete full loops
        for step in range(400):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if MODE == "train": np.save(Q_TABLE_FILE, q_table)
                    pygame.quit()
                    sys.exit()

            current_state = get_state(dot_x, dot_y)
            action = choose_action(current_state, EPSILON, q_table)
            
            # Calculate movement vector (dx, dy)
            dx, dy = 0, 0
            if action == 0: dy = -DOT_SPEED  # Up
            elif action == 1: dy = DOT_SPEED # Down
            elif action == 2: dx = -DOT_SPEED # Left
            elif action == 3: dx = DOT_SPEED # Right

            # Apply Move
            dot_x += dx
            dot_y += dy

            # Boundary check
            dot_x = max(DOT_RADIUS, min(SCREEN_WIDTH - DOT_RADIUS, dot_x))
            dot_y = max(DOT_RADIUS, min(SCREEN_HEIGHT - DOT_RADIUS, dot_y))

            # REWARD LOGIC (Calculated AFTER move)
            new_state = get_state(dot_x, dot_y)
            
            # Pass the movement vector (dx, dy) to the reward function
            reward = get_reward(dot_x, dot_y, dx, dy)
            
            if MODE == "train":
                old_q = q_table[current_state[0], current_state[1], action]
                max_future_q = np.max(q_table[new_state[0], new_state[1], :])
                new_q = old_q + ALPHA * (reward + GAMMA * max_future_q - old_q)
                q_table[current_state[0], current_state[1], action] = new_q

            # Visualization (Draw every frame in run, skip in train)
            if MODE == "run" or episode % 100 == 0:
                screen.fill(BLACK)
                # pygame.draw.circle(screen, GREEN, CIRCLE_CENTER, CIRCLE_RADIUS, 2)
                pygame.draw.circle(screen, RED, (int(dot_x), int(dot_y)), DOT_RADIUS)
                pygame.display.flip()
                
                if MODE == "run": clock.tick(60)
        
        if MODE == "train":
            EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
            if episode % 100 == 0:
                print(f"Episode {episode}, Epsilon: {EPSILON:.4f}")

    if MODE == "train":
        np.save(Q_TABLE_FILE, q_table)
        print("Training Saved.")

if __name__ == "__main__":
    main()