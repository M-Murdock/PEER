import pygame
import sys
import numpy as np
import math

# --- Constants ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

DOT_RADIUS = 8
DOT_SPEED = 5 

# --- Triangle Target Definition ---
# Define the 3 corners (Vertices)
V1 = np.array([300, 100]) # Top
V2 = np.array([100, 500]) # Bottom Left
V3 = np.array([500, 500]) # Bottom Right

# Define the 3 edges (Start Point -> End Point)
# The order dictates the direction of flow (Clockwise or Counter-Clockwise)
EDGES = [
    (V1, V2), # Edge 1: Down-Left
    (V2, V3), # Edge 2: Right
    (V3, V1)  # Edge 3: Up-Left
]

Q_TABLE_FILE = "q_table_triangle.npy"

# --- Q-Learning Parameters ---
GRID_SIZE = 20 
STATES_X = SCREEN_WIDTH // GRID_SIZE 
STATES_Y = SCREEN_HEIGHT // GRID_SIZE 

MODE = "run" 

def get_state(x, y):
    state_x = int(max(0, min(x, SCREEN_WIDTH - 1)) // GRID_SIZE)
    state_y = int(max(0, min(y, SCREEN_HEIGHT - 1)) // GRID_SIZE)
    return (state_x, state_y)

def get_distance_and_flow(p, start, end):
    """
    Calculates distance from point p to line segment (start-end).
    Also returns the normalized direction vector of the segment.
    """
    # Vector of the line segment
    line_vec = end - start
    # Vector from start to point
    point_vec = p - start
    
    # Length of the line segment squared
    line_len_sq = np.dot(line_vec, line_vec)
    
    if line_len_sq == 0:
        return np.linalg.norm(p - start), (0,0) # Start and end are same

    # Project point onto line (parameter t)
    # t represents where on the line the projection falls (0 to 1)
    t = np.dot(point_vec, line_vec) / line_len_sq
    t = max(0, min(1, t)) # Clamp to segment
    
    # Find the closest point on the segment
    projection = start + t * line_vec
    
    # Distance from agent to that projection
    dist = np.linalg.norm(p - projection)
    
    # Flow Vector: The unit vector pointing from start to end
    flow_vec = line_vec / np.sqrt(line_len_sq)
    
    return dist, flow_vec

def get_reward(x, y, dx, dy):
    current_pos = np.array([float(x), float(y)])
    
    # --- 1. CORNER HANDLING (The Fix) ---
    # Check if we are close to any vertex. 
    # If so, force the flow to point to the NEXT vertex immediately.
    # Threshold: 20 pixels (approx 1 grid cell)
    CORNER_THRESHOLD = 20.0
    
    # V1 (Top) -> starts Edge 1
    if np.linalg.norm(current_pos - V1) < CORNER_THRESHOLD:
        best_flow = (V2 - V1) / np.linalg.norm(V2 - V1)
        best_dist = np.linalg.norm(current_pos - V1)
        
    # V2 (Bottom Left) -> starts Edge 2
    elif np.linalg.norm(current_pos - V2) < CORNER_THRESHOLD:
        best_flow = (V3 - V2) / np.linalg.norm(V3 - V2)
        best_dist = np.linalg.norm(current_pos - V2)

    # V3 (Bottom Right) -> starts Edge 3 (This fixes your stuck agent!)
    elif np.linalg.norm(current_pos - V3) < CORNER_THRESHOLD:
        best_flow = (V1 - V3) / np.linalg.norm(V1 - V3)
        best_dist = np.linalg.norm(current_pos - V3)

    # --- 2. STANDARD EDGE HANDLING ---
    else:
        # If not near a corner, do the standard "closest line" logic
        best_dist = float('inf')
        best_flow = np.array([0.0, 0.0])
        
        for start, end in EDGES:
            dist, flow = get_distance_and_flow(current_pos, start, end)
            if dist < best_dist:
                best_dist = dist
                best_flow = flow

    # --- 3. CALCULATE REWARD ---
    
    # Gravity: Penalty for being far from the line
    reward_position = -best_dist * 2.0 

    # Flow: Reward for moving in the direction of the best_flow
    move_mag = math.sqrt(dx**2 + dy**2)
    if move_mag > 0:
        norm_dx = dx / move_mag
        norm_dy = dy / move_mag
        alignment = (norm_dx * best_flow[0]) + (norm_dy * best_flow[1])
    else:
        alignment = 0

    # Increased flow weight slightly to encourage pushing through corners
    reward_flow = alignment * 40 

    return reward_position + reward_flow

def choose_action(state, epsilon, q_table):
    if np.random.rand() < epsilon:
        return np.random.randint(4) 
    else:
        return np.argmax(q_table[state[0], state[1], :])

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Q-Learning Triangle - {MODE.upper()}")
    clock = pygame.time.Clock()

    q_table = np.zeros((STATES_X, STATES_Y, 4))
    
    # Training Params
    ALPHA = 0.1      
    GAMMA = 0.9      
    EPSILON = 1.0    
    EPSILON_DECAY = 0.9995 
    MIN_EPSILON = 0.01
    NUM_EPISODES = 12000 

    if MODE == "run":
        try:
            q_table = np.load(Q_TABLE_FILE)
            EPSILON = 0
        except:
            print("Train first!")
            sys.exit()

    for episode in range(NUM_EPISODES):
        dot_x, dot_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        
        for step in range(400):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if MODE == "train": np.save(Q_TABLE_FILE, q_table)
                    pygame.quit()
                    sys.exit()

            current_state = get_state(dot_x, dot_y)
            action = choose_action(current_state, EPSILON, q_table)
            
            dx, dy = 0, 0
            if action == 0: dy = -DOT_SPEED  # Up
            elif action == 1: dy = DOT_SPEED # Down
            elif action == 2: dx = -DOT_SPEED # Left
            elif action == 3: dx = DOT_SPEED # Right

            dot_x += dx
            dot_y += dy

            # Boundary check
            dot_x = max(DOT_RADIUS, min(SCREEN_WIDTH - DOT_RADIUS, dot_x))
            dot_y = max(DOT_RADIUS, min(SCREEN_HEIGHT - DOT_RADIUS, dot_y))

            # REWARD LOGIC
            new_state = get_state(dot_x, dot_y)
            reward = get_reward(dot_x, dot_y, dx, dy)
            
            if MODE == "train":
                old_q = q_table[current_state[0], current_state[1], action]
                max_future_q = np.max(q_table[new_state[0], new_state[1], :])
                new_q = old_q + ALPHA * (reward + GAMMA * max_future_q - old_q)
                q_table[current_state[0], current_state[1], action] = new_q

            # Visualization
            if MODE == "run" or episode % 100 == 0:
                screen.fill(BLACK)
                
                # Draw Agent
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
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()