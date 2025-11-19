# Import the pygame library
import pygame
import sys
import numpy as np


class Dot_Simulator:
    def __init__(self):
            # --- Constants ---
        # Screen dimensions
        self.SCREEN_WIDTH = 600
        self.SCREEN_HEIGHT = 600

        # Colors (R, G, B)
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0) # For the target square
        self.RED = (255, 0, 0) # For the dot (to see it better)

        # Dot properties
        self.DOT_RADIUS = 8
        self.DOT_SPEED = 5 # This is now the agent's action magnitude

        # --- Top Left ---
        self.TOP_LEFT_X = 14
        self.TOP_LEFT_Y = 12
        self.Q_TABLE_FILE = "q_table_topleft.npy"
        # Discretize the state space
        self.GRID_SIZE = 20 # 20x20 pixel cells
        self.STATES_X = self.SCREEN_WIDTH // self.GRID_SIZE   # 30 states
        self.STATES_Y = self.SCREEN_HEIGHT // self.GRID_SIZE  # 30 states

        # --- Initialization ---
        # Initialize all imported pygame modules
        pygame.init()
        # Set up the display window
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Dot Mover Simulation")
        # Clock for controlling the frame rate (though not strictly needed for this event-based movement)
        self.clock = pygame.time.Clock()





    def get_state(self, x, y):
        """Converts (x, y) coordinates to a discrete grid state."""
        state_x = int(max(0, min(x, self.SCREEN_WIDTH - 1)) // self.GRID_SIZE)
        state_y = int(max(0, min(y, self.SCREEN_HEIGHT - 1)) // self.GRID_SIZE)
        return (state_x, state_y)

    def choose_action(self, state, epsilon, q_table):
        """Epsilon-greedy action selection."""
        if np.random.rand() < epsilon:
            return np.random.randint(4) # Explore: random action
        else:
            # Exploit: choose best action from Q-table
            return np.argmax(q_table[state[0], state[1], :])
    
    def run_auton(self):
        print("--- Mode: RUNNING POLICY ---")

        # --- Load Q-Table ---
        try:
            q_table = np.load(self.Q_TABLE_FILE)
            print(f"Loaded Q-table from {self.Q_TABLE_FILE}")
        except FileNotFoundError:
            print(f"Error: Q-table file '{self.Q_TABLE_FILE}' not found.")
            print("Please run the script in 'train' mode first.")
            pygame.quit()
            sys.exit()

        # --- Run Policy Loop ---
        dot_x = self.SCREEN_WIDTH // 2
        dot_y = self.SCREEN_HEIGHT // 2

        running = True
        while running:
            # --- Event Handling (to allow quitting) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # --- Agent Logic (Exploitation only) ---
            # 1. Get current state
            current_state = self.get_state(dot_x, dot_y)
            
            # 2. Choose action (epsilon=0 for pure exploitation)
            action = self.choose_action(current_state, 0, q_table)
            
            # 3. Take action (move the dot)
            if action == 0: # Up
                dot_y -= self.DOT_SPEED
            elif action == 1: # Down
                dot_y += self.DOT_SPEED
            elif action == 2: # Left
                dot_x -= self.DOT_SPEED
            elif action == 3: # Right
                dot_x += self.DOT_SPEED

            # 4. Boundary check
            dot_x = max(self.DOT_RADIUS, min(self.SCREEN_WIDTH - self.DOT_RADIUS, dot_x))
            dot_y = max(self.DOT_RADIUS, min(self.SCREEN_HEIGHT - self.DOT_RADIUS, dot_y))

            # 5. NO reward or Q-table update in 'run' mode

            # --- Drawing (every frame) ---
            self.screen.fill(self.BLACK)
            pygame.draw.circle(self.screen, self.GREEN, (self.TOP_LEFT_X, self.TOP_LEFT_Y), self.DOT_RADIUS)
            pygame.draw.circle(self.screen, self.RED, (int(dot_x), int(dot_y)), self.DOT_RADIUS)
            
            # --- Update Display ---
            pygame.display.flip()
            self.clock.tick(60) # Run at a viewable speed

        pygame.quit()
        sys.exit()

    def run_teleop(self):
        dot_x = self.SCREEN_WIDTH // 2
        dot_y = self.SCREEN_HEIGHT // 2
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
                        dot_y -= self.DOT_SPEED
                    elif event.key == pygame.K_DOWN:
                        dot_y += self.DOT_SPEED
                    elif event.key == pygame.K_LEFT:
                        dot_x -= self.DOT_SPEED
                    elif event.key == pygame.K_RIGHT:
                        dot_x += self.DOT_SPEED
            # --- Game Logic ---
            # Boundary check to keep the dot on the screen

            # --- Drawing ---
            # Fill the entire screen with black
            self.screen.fill(self.BLACK)

            # Draw the white dot on the screen
            # pygame.draw.circle(surface, color, center_pos, radius)
            pygame.draw.circle(self.screen, self.WHITE, (dot_x, dot_y), self.DOT_RADIUS)

            # --- Update Display ---
            # Flip the display to show the new frame
            pygame.display.flip()

            # (Optional) Cap the frame rate
            self.clock.tick(60)
            
    def run_shared(self):
        pass



print("running")
# print(run())   
# run_auton() 
dot = Dot_Simulator()
# dot.run_teleop()
dot.run_auton()

