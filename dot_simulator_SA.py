# Import the pygame library
import pygame
import sys
import numpy as np
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

# --- Top Left ---
TOP_LEFT_X = 14
TOP_LEFT_Y = 12
Q_TABLE_FILE = "q_table_topleft.npy"
# Discretize the state space
GRID_SIZE = 20 # 20x20 pixel cells
STATES_X = SCREEN_WIDTH // GRID_SIZE   # 30 states
STATES_Y = SCREEN_HEIGHT // GRID_SIZE  # 30 states

# --- Initialization ---
# Initialize all imported pygame modules
pygame.init()
# Set up the display window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Dot Mover Simulation")
# Clock for controlling the frame rate (though not strictly needed for this event-based movement)
clock = pygame.time.Clock()






def get_state(x, y):
    """Converts (x, y) coordinates to a discrete grid state."""
    state_x = int(max(0, min(x, SCREEN_WIDTH - 1)) // GRID_SIZE)
    state_y = int(max(0, min(y, SCREEN_HEIGHT - 1)) // GRID_SIZE)
    return (state_x, state_y)

def choose_action(state, epsilon, q_table):
    """Epsilon-greedy action selection."""
    if np.random.rand() < epsilon:
        return np.random.randint(4) # Explore: random action
    else:
        # Exploit: choose best action from Q-table
        return np.argmax(q_table[state[0], state[1], :])
    
def run_auton():
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

    pygame.quit()
    sys.exit()

def run_teleop():
    dot_x = SCREEN_WIDTH // 2
    dot_y = SCREEN_HEIGHT // 2
    # --- Main Game Loop ---
    running = True
    while running:
        # --- Event Handling ---
        # Check for all user events in the queue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # If the user clicks the window's close button
                running = False
            elif event.type == pygame.KEYDOWN:
                # If a key is pressed
                if event.key == pygame.K_UP:
                    dot_y -= DOT_SPEED
                elif event.key == pygame.K_DOWN:
                    dot_y += DOT_SPEED
                elif event.key == pygame.K_LEFT:
                    dot_x -= DOT_SPEED
                elif event.key == pygame.K_RIGHT:
                    dot_x += DOT_SPEED
        # --- Game Logic ---
        # Boundary check to keep the dot on the screen

        # --- Drawing ---
        # Fill the entire screen with black
        screen.fill(BLACK)

        # Draw the white dot on the screen
        # pygame.draw.circle(surface, color, center_pos, radius)
        pygame.draw.circle(screen, WHITE, (dot_x, dot_y), DOT_RADIUS)

        # --- Update Display ---
        # Flip the display to show the new frame
        pygame.display.flip()

        # (Optional) Cap the frame rate
        clock.tick(60)

print("running")
# print(run())   
# run_auton() 
run_teleop() 
# --- Shutdown ---
# Once the loop is exited, quit pygame and the program
