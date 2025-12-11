# Import the pygame library
import pygame
import pygame.freetype
import sys
import numpy as np
from util.shared_auto import SharedAutoPolicy
from util.predictors import BayesianPredictor, MaxEntPredictor, CRFPredictor
from os import listdir
from os.path import isfile, join
from util.selector import Method_Selector
from util.SA_types import Inference, Assistance, Arbitration
from training import policy_drawing_correspondences

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

# -------------------------------------------------------------------------------------------------------------------
class Dot_Simulator:
    def __init__(self, policy_dir="trained_policies", inference_type=Inference.BAYESIAN, assistance_type=Assistance.DISTRIBUTION, arbitration_type=Arbitration.LINEAR):

            # --- Constants ---
        self.GAMMA = 0.4
        # Screen dimensions
        self.SCREEN_WIDTH = 600
        self.SCREEN_HEIGHT = 600
        self.TOP_PANEL_HEIGHT = 175   # space above the dot region

        # Checkboxes for enabling/disabling visualizations
        self.prob_visualization_on = True
        self.CHECKBOX_SIZE = 20
        self.PROB_CHECKBOX_POS = (10, self.TOP_PANEL_HEIGHT - self.CHECKBOX_SIZE - 140)
        
        self.goal_visualization_on = True
        self.GOAL_CHECKBOX_POS = (200, self.TOP_PANEL_HEIGHT - self.CHECKBOX_SIZE - 140)


        # Colors 
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)

        # Dot properties
        self.DOT_RADIUS = 8
        self.DOT_SPEED = 5 # This is now the agent's action magnitude

        # Discretize the state space
        self.GRID_SIZE = 20 # 20x20 pixel cells
        self.STATES_X = self.SCREEN_WIDTH // self.GRID_SIZE   # 30 states
        self.STATES_Y = self.SCREEN_HEIGHT // self.GRID_SIZE  # 30 states
        
        self.dot_x = self.SCREEN_WIDTH // 2
        self.dot_y = self.SCREEN_HEIGHT // 2

        # up, down, left, right
        self.ACTION_SPACE_LEN = 4
        
        self.INFERENCE_TYPE = inference_type
        self.ASSISTANCE_TYPE = assistance_type
        self.ARBITRATION_TYPE = arbitration_type

        # checkbox settings
        self.last_click_time = 0
        self.CLICK_COOLDOWN = 50  # ms
        
            # --- Initialization ---
        # Initialize all imported pygame modules
        pygame.init()
        # Set up the display window
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT + self.TOP_PANEL_HEIGHT))

        caption = "Dot Mover Simulation: " + inference_type.value + ", " + assistance_type.value + ", " + arbitration_type.value
        pygame.display.set_caption(caption)
        # set the font
        self.TEXT_SIZE = 15
        pygame.font.init() 
        self.font = pygame.freetype.SysFont('Arial', self.TEXT_SIZE)
        # Clock for controlling the frame rate 
        self.clock = pygame.time.Clock()
        
        # Get all the policies from the given directory
        self.POLICY_DIR = policy_dir
        self.POLICY_FILES = [f for f in listdir(self.POLICY_DIR) if isfile(join(self.POLICY_DIR, f))]
        self.POLICIES = [Dot_Policy(pi) for pi in [join(self.POLICY_DIR, f) for f in self.POLICY_FILES]]
        
        self.prob = np.zeros(len(self.POLICIES))
        # colors that correspond with each policy
        self.POLICY_COLORS = self.generate_colors(n=len(self.POLICIES))
        
        
    def generate_colors(self, n=1):
        colors = []
        for i in range(n):
            # use scaling to spread values evenly in 1–254
            r = ((i+1) * 123) % 254
            g = ((i+1) * 231) % 254
            b = ((i+1) * 77) % 254
            colors.append((r, g, b))
        return colors

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
    
    def execute_action(self, action): # note: action must be in form: (x, y)   
        self.dot_x += self.DOT_SPEED * action[0] # execute the action
        self.dot_y += self.DOT_SPEED * action[1]
    
    def ensure_within_boundaries(self):
        self.dot_x = max(self.DOT_RADIUS, min(self.SCREEN_WIDTH - self.DOT_RADIUS, self.dot_x))
        self.dot_y = max(self.DOT_RADIUS, min(self.SCREEN_HEIGHT - self.DOT_RADIUS, self.dot_y)) 

    def draw_probability_bars(self):
        """
        Draws a horizontal bar chart for the goal probabilities in the top panel.
        """
        num_policies = len(self.prob)
        bar_width = self.SCREEN_WIDTH // num_policies
        max_bar_height = self.TOP_PANEL_HEIGHT - 75  # space for labels

        # --- Distinct background for the top panel ---
        pygame.draw.rect(
            self.screen, 
            (25, 25, 25), 
            pygame.Rect(0, 0, self.SCREEN_WIDTH, self.TOP_PANEL_HEIGHT)
        )

        # --- Draw a separator line to clearly divide top panel from gameplay ---
        pygame.draw.line(
            self.screen,
            (180, 180, 180),
            (0, self.TOP_PANEL_HEIGHT - 2),
            (self.SCREEN_WIDTH, self.TOP_PANEL_HEIGHT - 2),
            3
        )

        for i, p in enumerate(self.prob):
            # Bar geometry
            x = i * bar_width
            bar_x = x + 8
            bar_w = bar_width - 16

            filled_height = int(max_bar_height * float(p))
            empty_height = max_bar_height - filled_height
            bar_bottom_y = self.TOP_PANEL_HEIGHT - 5

            # --- Background track (for contrast) ---
            pygame.draw.rect(
                self.screen,
                (80, 80, 80),
                pygame.Rect(bar_x, bar_bottom_y - max_bar_height, bar_w, max_bar_height)
            )

            # --- Filled probability bar --- 
            pygame.draw.rect(
                self.screen,
                self.POLICY_COLORS[i],
                pygame.Rect(bar_x, bar_bottom_y - filled_height, bar_w, filled_height)
            )

            # --- Policy label and percentage ---
            label = f"{i}: {p*100:.1f}%"
            text_surface, _ = self.font.render(label, self.WHITE)
            self.screen.blit(text_surface, (bar_x, 50))
            
    def draw_checkbox(self):
        # Checkbox for goal visualization
        # Draw checkbox outline
        pygame.draw.rect(
            self.screen,
            self.WHITE,
            pygame.Rect(self.GOAL_CHECKBOX_POS[0], self.GOAL_CHECKBOX_POS[1], self.CHECKBOX_SIZE, self.CHECKBOX_SIZE),
            2
        )

        # Draw checkmark if probability visualization is on
        if self.goal_visualization_on:
            pygame.draw.rect(
            self.screen,
            self.WHITE,
            pygame.Rect(self.GOAL_CHECKBOX_POS[0], self.GOAL_CHECKBOX_POS[1], self.CHECKBOX_SIZE, self.CHECKBOX_SIZE, width=0)
        )


        # Checkbox label
        label_surface, _ = self.font.render("Show Goals", self.WHITE)
        self.screen.blit(label_surface, (self.GOAL_CHECKBOX_POS[0] + self.CHECKBOX_SIZE + 5, self.GOAL_CHECKBOX_POS[1] - 2))
        
        # ----------------------------
        # Checkbox for probabilities
        # Draw checkbox outline
        pygame.draw.rect(
            self.screen,
            self.WHITE,
            pygame.Rect(self.PROB_CHECKBOX_POS[0], self.PROB_CHECKBOX_POS[1], self.CHECKBOX_SIZE, self.CHECKBOX_SIZE),
            2
        )

        # Draw checkmark if probability visualization is on
        if self.prob_visualization_on:
            pygame.draw.rect(
            self.screen,
            self.WHITE,
            pygame.Rect(self.PROB_CHECKBOX_POS[0], self.PROB_CHECKBOX_POS[1], self.CHECKBOX_SIZE, self.CHECKBOX_SIZE, width=0)
        )

        # Checkbox label
        label_surface, _ = self.font.render("Show Probabilities", self.WHITE)
        self.screen.blit(label_surface, (self.PROB_CHECKBOX_POS[0] + self.CHECKBOX_SIZE + 5, self.PROB_CHECKBOX_POS[1] - 2))


    def draw_goal_visualizations(self):
        # for each file in trained_policies
        for i, f in enumerate(self.POLICY_FILES):
            # get the corresponding visualization info from visualization_correspondences.csv
            draw_data = policy_drawing_correspondences.get_data_by_filename(f)
            
            if not draw_data is None:
            
                if draw_data["type"] == "circle":
                    # draw the visualization
                    pygame.draw.circle(self.screen, self.POLICY_COLORS[i], (float(draw_data["x"]), float(draw_data["y"])+self.TOP_PANEL_HEIGHT), float(draw_data["r"]), width=1)
                if draw_data["type"] == "point":
                    pygame.draw.circle(self.screen, self.POLICY_COLORS[i], (float(draw_data["x"]), float(draw_data["y"])+self.TOP_PANEL_HEIGHT), self.DOT_RADIUS)      
        
    def redraw_screen(self):
        # Fill bottom (gameplay region) with a different background for clear separation
        self.screen.fill((0, 0, 0))

        # 1. Top panel
        # self.draw_probability_bars()
        if self.prob_visualization_on:
            self.draw_probability_bars()
        if self.goal_visualization_on:
            self.draw_goal_visualizations()

        # 2. Gameplay area (below the panel)
        dot_y = self.dot_y + self.TOP_PANEL_HEIGHT
        pygame.draw.circle(self.screen, self.WHITE, (self.dot_x, dot_y), self.DOT_RADIUS)
        
        self.draw_checkbox()
        
        pygame.display.flip()
        self.clock.tick(60)
            
            
    def run_shared(self):

        # Get the selected inference method
        if self.INFERENCE_TYPE is Inference.BAYESIAN: # Bayesian Prediction
            pred = BayesianPredictor(self.POLICIES)
        elif self.INFERENCE_TYPE is Inference.MAX_ENT: # Max Entropy Prediction
            pred = MaxEntPredictor(self.POLICIES)
        elif self.INFERENCE_TYPE is Inference.CRF: # Conditional Random Field Prediction
            pred = CRFPredictor(self.POLICIES)

            
        # Get the selected assistance method
        if self.ASSISTANCE_TYPE is Assistance.DISTRIBUTION:
            policy = SharedAutoPolicy(self.POLICIES, list(range(self.ACTION_SPACE_LEN)))
        else:
            pass
        
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
            if keys[pygame.K_DOWN]:
                u = 1
            if keys[pygame.K_LEFT]:
                u = 2
            if keys[pygame.K_RIGHT]:
                u = 3
                
            # don't move the dot until AFTER the user has given their first control signal
            if u == -1:
                self.redraw_screen()
                continue
            
            # Inference: get the probability of the policies based on the user's control signal
            self.prob = pred.update(self.get_state(self.dot_x, self.dot_y), u)

            # Assistance: using the most likely policy, calculate the next optimal action
            optimal_action = policy.get_action(self.get_state(self.dot_x, self.dot_y), self.prob) # Get robot's predicted action

            # Arbitration: blend the optimal action and control signal
            self.execute_action(self.blend(u, optimal_action))

            # Boundary check to keep the dot on the screen
            self.ensure_within_boundaries()
            # Redraw the dot in its new position and display probabilities
            self.redraw_screen()
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                now = pygame.time.get_ticks()
                if now - self.last_click_time < self.CLICK_COOLDOWN:
                    continue
                self.last_click_time = now

                mouse_pos = event.pos

                checkbox_rect = pygame.Rect(self.PROB_CHECKBOX_POS, (self.CHECKBOX_SIZE, self.CHECKBOX_SIZE))
                goal_checkbox_rect = pygame.Rect(self.GOAL_CHECKBOX_POS, (self.CHECKBOX_SIZE, self.CHECKBOX_SIZE))

                if checkbox_rect.collidepoint(mouse_pos):
                    self.prob_visualization_on = not self.prob_visualization_on

                if goal_checkbox_rect.collidepoint(mouse_pos):
                    self.goal_visualization_on = not self.goal_visualization_on


            
    # compute an action which blends u and a*
    def blend(self, u, a):
        # convert from action index to x,y
        u = self.index_to_tuple(u)
        a = self.index_to_tuple(a)
        
        # ---------------------------------
        # linear arbitration
        if self.ARBITRATION_TYPE is Arbitration.LINEAR: 
            blended = ((u[0]*self.GAMMA) + (a[0]*(1-self.GAMMA)), (u[1]*self.GAMMA) + (a[1]*(1-self.GAMMA)))
            # convert to unit vector
            mag = np.sqrt(blended[0]*blended[0] + blended[1]*blended[1])
            if mag == 0:
                return (0,0)
            return (blended[0]/mag, blended[1]/mag)
        
        # ---------------------------------
        # probabilistic arbitration
        elif self.ARBITRATION_TYPE is Arbitration.PROBABILISTIC:
            self.robot_confidence = max(self.prob)
            p_robot = float(self.robot_confidence)
            # Probabilistic mixture
            blended = (
                p_robot * a[0] + (1 - p_robot) * u[0],
                p_robot * a[1] + (1 - p_robot) * u[1]
            )
            # Normalize
            mag = np.sqrt(blended[0]**2 + blended[1]**2)
            if mag == 0:
                return (0, 0)
            return (blended[0]/mag, blended[1]/mag)
        
        # ---------------------------------
        # no blending (i.e. don't follow user input at all)
        elif self.ARBITRATION_TYPE is Arbitration.ONLY_ROBOT:
            return a
        # ---------------------------------


# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# get the inference method from the user
inference_selector = Method_Selector(options=[i for i in Inference], caption="Inference Method")
inference_type = inference_selector.get() # get the user's selection 

# get the assistance method from the user
# assistance_selector = Method_Selector(options=[d for d in Assistance], caption="Assistance Method")
# assistance_type = assistance_selector.get() # get the user's selection 

# get the blending method from the user
arbitration_selector = Method_Selector(options=[a for a in Arbitration], caption="Arbitration Method")
arbitration_type = arbitration_selector.get() # get the user's selection 


# dot = Dot_Simulator(inference_type=inference_type, assistance_type=assistance_type, arbitration_type=arbitration_type)
dot = Dot_Simulator(inference_type=inference_type, arbitration_type=arbitration_type)
dot.run_shared()

