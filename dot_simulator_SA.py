# Import the pygame library
import pygame
import sys
import numpy as np
from shared_auto import SharedAutoPolicy
from maxent_pred import MaxEntPredictor

class Dot_Policy:
    def __init__(self, q_table_file="q_table_topleft.npy"):
        self.q_table_file = q_table_file
        # --- Load Q-Table ---
        try:
            self.q_table = np.load(self.q_table_file)
            print(f"Loaded Q-table from {self.q_table_file}")
        except FileNotFoundError:
            print(f"Error: Q-table file '{self.q_table_file}' not found.")
    
    def get_q_value(self, state, action): 
        # return q value for a given action
        print(self.q_table[state[0], state[1], action])

        return self.q_table[state[0], state[1], action]

    # def get_action_indices(self, dimension, value):
    #     indices = []
    #     for action_index in range(0, len(self.actions)):
    #         if self.actions[action_index][dimension] == value:
    #             indices.append(action_index)
    #     return indices

    def get_action(self, state):
        return np.argmax(self.q_table[state[0], state[1], :])

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
        
        self.dot_x = self.SCREEN_WIDTH // 2
        self.dot_y = self.SCREEN_HEIGHT // 2

        # self.ACTIONS = [(), 1, 2, 3] # up, down, left, right
        self.ACTION_SPACE_LEN = 4

        # --- Initialization ---
        # Initialize all imported pygame modules
        pygame.init()
        # Set up the display window
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Dot Mover Simulation")
        # Clock for controlling the frame rate (though not strictly needed for this event-based movement)
        self.clock = pygame.time.Clock()
        
        self.POLICIES = [Dot_Policy(self.Q_TABLE_FILE)]


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

        test_policy = self.POLICIES[0]
        running = True
        while running:
            # --- Event Handling (to allow quitting) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # --- Agent Logic (Exploitation only) ---
            # 1. Get current state
            current_state = self.get_state(self.dot_x, self.dot_y)
            
            # 2. Choose action (epsilon=0 for pure exploitation)
            action = test_policy.get_action(current_state)
            
            # 3. Take action (move the dot)
            if action == 0: # Up
                self.dot_y -= self.DOT_SPEED
            elif action == 1: # Down
                self.dot_y += self.DOT_SPEED
            elif action == 2: # Left
                self.dot_x -= self.DOT_SPEED
            elif action == 3: # Right
                self.dot_x += self.DOT_SPEED

            # 4. Boundary check
            self.dot_x = max(self.DOT_RADIUS, min(self.SCREEN_WIDTH - self.DOT_RADIUS, self.dot_x))
            self.dot_y = max(self.DOT_RADIUS, min(self.SCREEN_HEIGHT - self.DOT_RADIUS, self.dot_y))

            # --- Drawing (every frame) ---
            self.screen.fill(self.BLACK)
            pygame.draw.circle(self.screen, self.GREEN, (self.TOP_LEFT_X, self.TOP_LEFT_Y), self.DOT_RADIUS)
            pygame.draw.circle(self.screen, self.RED, (int(self.dot_x), int(self.dot_y)), self.DOT_RADIUS)
            
            # --- Update Display ---
            pygame.display.flip()
            self.clock.tick(60) # Run at a viewable speed

        pygame.quit()
        sys.exit()

    def run_teleop(self):
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
                        self.dot_y -= self.DOT_SPEED
                    elif event.key == pygame.K_DOWN:
                        self.dot_y += self.DOT_SPEED
                    elif event.key == pygame.K_LEFT:
                        self.dot_x -= self.DOT_SPEED
                    elif event.key == pygame.K_RIGHT:
                        self.dot_x += self.DOT_SPEED
            # --- Game Logic ---
            # Boundary check to keep the dot on the screen
            # 4. Boundary check
            self.dot_x = max(self.DOT_RADIUS, min(self.SCREEN_WIDTH - self.DOT_RADIUS, self.dot_x))
            self.dot_y = max(self.DOT_RADIUS, min(self.SCREEN_HEIGHT - self.DOT_RADIUS, self.dot_y))
            # --- Drawing ---
            # Fill the entire screen with black
            self.screen.fill(self.BLACK)

            # Draw the white dot on the screen
            # pygame.draw.circle(surface, color, center_pos, radius)
            pygame.draw.circle(self.screen, self.WHITE, (self.dot_x, self.dot_y), self.DOT_RADIUS)

            # --- Update Display ---
            # Flip the display to show the new frame
            pygame.display.flip()

            # (Optional) Cap the frame rate
            self.clock.tick(60)
            
    def run_shared(self):
        # pass
        pred = MaxEntPredictor(self.POLICIES)
        # policy = SharedAutoPolicy(policies, list(range(len(action_space))))
        policy = SharedAutoPolicy(self.POLICIES, list(range(self.ACTION_SPACE_LEN)))
        
        
        u = -1
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
                        self.dot_y -= self.DOT_SPEED
                        u = 0
                    elif event.key == pygame.K_DOWN:
                        self.dot_y += self.DOT_SPEED
                        u = 1
                    elif event.key == pygame.K_LEFT:
                        self.dot_x -= self.DOT_SPEED
                        u = 2
                    elif event.key == pygame.K_RIGHT:
                        self.dot_x += self.DOT_SPEED
                        u = 3
                        
            # prob = pred.get_prob_after_obs(self.get_state(), u)
            prob = pred.update(self.get_state(self.dot_x, self.dot_y), u)
            optimal_action = policy.get_action(self.get_state(self.dot_x, self.dot_y), prob)# Get robot's predicted action
            blended_action = 1 # PLACEHOLDER
            print(f"Prob:{prob}, Optimal Action:{optimal_action}")
            
            # q_all = [policy.get_q_value(state, u_h_index) for policy in policies]
            # best_actions = [action_space[policy.get_action(state)] for policy in policies]

            # print(f"{state} -> {u_h} -> {best_actions} -> {q_all} -> {prob} -> {action_space[u_r_index]}") # For debugging
            
            # --- Game Logic ---
            # Boundary check to keep the dot on the screen
            # 4. Boundary check
            self.dot_x = max(self.DOT_RADIUS, min(self.SCREEN_WIDTH - self.DOT_RADIUS, self.dot_x))
            self.dot_y = max(self.DOT_RADIUS, min(self.SCREEN_HEIGHT - self.DOT_RADIUS, self.dot_y))
            # --- Drawing ---
            # Fill the entire screen with black
            self.screen.fill(self.BLACK)

            # Draw the white dot on the screen
            # pygame.draw.circle(surface, color, center_pos, radius)
            pygame.draw.circle(self.screen, self.WHITE, (self.dot_x, self.dot_y), self.DOT_RADIUS)

            # --- Update Display ---
            # Flip the display to show the new frame
            pygame.display.flip()

            # (Optional) Cap the frame rate
            self.clock.tick(60)


print("running")

dot = Dot_Simulator()
# dot.run_teleop()
# dot.run_auton()
dot.run_shared()

