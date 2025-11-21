# Import the pygame library
import pygame
import pygame.freetype
import sys
import numpy as np
from util.shared_auto import SharedAutoPolicy
from util.maxent_pred import MaxEntPredictor
from os import listdir
from os.path import isfile, join

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
        return self.q_table[state[0], state[1], action]

    def get_action(self, state):
        return np.argmax(self.q_table[state[0], state[1], :])

class Dot_Simulator:
    def __init__(self, policy_dir="trained_policies"):
            # --- Constants ---
        self.GAMMA = 0.4
        # Screen dimensions
        self.SCREEN_WIDTH = 600
        self.SCREEN_HEIGHT = 600

        # Colors 
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        # self.GREEN = (0, 255, 0) # For the target square
        self.RED = (255, 0, 0) # For the dot (to see it better)

        # Dot properties
        self.DOT_RADIUS = 8
        self.DOT_SPEED = 5 # This is now the agent's action magnitude

        # --- Top Left ---
        # Discretize the state space
        self.GRID_SIZE = 20 # 20x20 pixel cells
        self.STATES_X = self.SCREEN_WIDTH // self.GRID_SIZE   # 30 states
        self.STATES_Y = self.SCREEN_HEIGHT // self.GRID_SIZE  # 30 states
        
        self.dot_x = self.SCREEN_WIDTH // 2
        self.dot_y = self.SCREEN_HEIGHT // 2

        # up, down, left, right
        self.ACTION_SPACE_LEN = 4
        # len(self.q_table[state[0], state[1], :])

        # --- Initialization ---
        # Initialize all imported pygame modules
        pygame.init()
        # Set up the display window
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Dot Mover Simulation")
        # set the font
        self.TEXT_SIZE = 15
        pygame.font.init() 
        self.font = pygame.freetype.SysFont('Arial', self.TEXT_SIZE)
        # Clock for controlling the frame rate 
        self.clock = pygame.time.Clock()
        
        # Get all the policies from the given directory
        self.POLICY_DIR = policy_dir
        self.POLICIES = [Dot_Policy(pi) for pi in [join(self.POLICY_DIR, f) for f in listdir(self.POLICY_DIR) if isfile(join(self.POLICY_DIR, f))]]
        

    def get_state(self, x, y):
        """Converts (x, y) coordinates to a discrete grid state."""
        state_x = int(max(0, min(x, self.SCREEN_WIDTH - 1)) // self.GRID_SIZE)
        state_y = int(max(0, min(y, self.SCREEN_HEIGHT - 1)) // self.GRID_SIZE)
        return (state_x, state_y)

        
    def index_to_tuple(self, index):
        if index == 0:
            return (0, -1)
        elif index == 1:
            return (0, 1)
        elif index == 2:
            return (-1, 0)
        elif index == 3:
            return (1, 0)
        
        return (0, 0)
    
    def execute_action(self, action, is_tuple=False): # note: action must be in form: (x, y)
        if is_tuple == False: # if the action is NOT of form (x, y) then we need to convert it
            action = self.index_to_tuple(action)
            
        self.dot_x += self.DOT_SPEED * action[0] # execute the action
        self.dot_y += self.DOT_SPEED * action[1]
    
    def ensure_within_boundaries(self):
        self.dot_x = max(self.DOT_RADIUS, min(self.SCREEN_WIDTH - self.DOT_RADIUS, self.dot_x))
        self.dot_y = max(self.DOT_RADIUS, min(self.SCREEN_HEIGHT - self.DOT_RADIUS, self.dot_y))
            
    def redraw_screen(self, text=None):
        # Fill the entire screen with black
        self.screen.fill(self.BLACK)
        # Draw the white dot on the screen
        pygame.draw.circle(self.screen, self.WHITE, (self.dot_x, self.dot_y), self.DOT_RADIUS)
        # If there's any text we want to show (optional)
        if text:
            text_surface, _ = self.font.render(text, (255, 255, 255))
            self.screen.blit(text_surface, (5,5))
        # Flip the display to show the new frame
        pygame.display.flip()
        # Cap the frame rate
        self.clock.tick(60)
            
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

            # Boundary check to keep the dot on the screen
            self.ensure_within_boundaries()
            # Redraw the dot in its new position
            self.redraw_screen()

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
            
            # Boundary check to keep the dot on the screen
            self.ensure_within_boundaries()
            # Redraw the dot in its new position
            self.redraw_screen()
            
    def run_shared(self):

        pred = MaxEntPredictor(self.POLICIES)
        policy = SharedAutoPolicy(self.POLICIES, list(range(self.ACTION_SPACE_LEN)))
        
        u = -1
        running = True

        while running:
            # --- 1. Event Handling (Only for QUIT, initial key presses, and key releases) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                u = 0 
                self.execute_action(u)
            if keys[pygame.K_DOWN]:
                u = 1
                self.execute_action(u)
            if keys[pygame.K_LEFT]:
                u = 2
                self.execute_action(u)
            if keys[pygame.K_RIGHT]:
                u = 3
                self.execute_action(u)
                
            # don't move the dot until AFTER the user has given their first control signal
            if u == -1:
                self.redraw_screen()
                continue
            
            # get the probability of the policies based on the user's control signal
            if sum(keys) == 0: # no key is being pressed:
                prob = pred.get_prob()
                # prob = pred.get_prob_after_obs(self.get_state(self.dot_x, self.dot_y), u)
            else:
                prob = pred.get_prob_after_obs(self.get_state(self.dot_x, self.dot_y), u)
                
            # using the most likely policy, calculate the next optimal action
            optimal_action = policy.get_action(self.get_state(self.dot_x, self.dot_y), prob) # Get robot's predicted action
            # print(f"Prob:{prob}, Optimal Action:{optimal_action}")

            # using the optimal action and control signal, blend them together
            if u == optimal_action: # if the control signal and optimal action are the same, just execute it     
                self.execute_action(u)
                    
            else: 
                blended_action = self.blend(self.index_to_tuple(u),  self.index_to_tuple(optimal_action))
                # blended_action = [self.blend(a,b) for a, b in zip(self.index_to_tuple(u), self.index_to_tuple(optimal_action))]
                # print("BLENDED")
                # print(f"u: {self.index_to_tuple(u)}, a: {self.index_to_tuple(optimal_action)}, blended: {blended_action}")
                self.execute_action(blended_action, is_tuple=True)

            # Boundary check to keep the dot on the screen
            self.ensure_within_boundaries()
            # Redraw the dot in its new position
            self.redraw_screen(f"Prob: {prob}")
            
    # compute an action which blends u and a*
    def blend(self, u, a):
        # return (a*self.GAMMA) + (b*(1-self.GAMMA))
        # action = gamma*u + (1-gamma)a
        return ((u[0]*self.GAMMA) + (a[0]*(1-self.GAMMA)), (u[1]*self.GAMMA) + (a[1]*(1-self.GAMMA)))


dot = Dot_Simulator()
# dot.run_teleop()
# dot.run_auton()
dot.run_shared()

