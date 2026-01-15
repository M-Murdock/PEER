# Refactored Dot Simulator (behavior preserved)
import sys
import os
from os import listdir
from os.path import isfile, join

import numpy as np
import pygame
import pygame.freetype

from util.shared_auto import SharedAutoPolicy
from util.predictors import BayesianPredictor, MaxEntPredictor, CRFPredictor
from util.selector import Method_Selector
from util.SA_types import Inference, Assistance, Arbitration
from training import policy_drawing_correspondences


# -------------------------
# Lightweight Q-table wrapper
# -------------------------
class Dot_Policy:
    """Simple adapter around a saved Q-table .npy file.

    Public API:
        get_q_value(state, action)
        get_action(state) -> argmax over actions
    """
    def __init__(self, q_table_file="q_table_topleft.npy"):
        self.q_table_file = q_table_file
        try:
            self.q_table = np.load(self.q_table_file)
            print(f"Loaded Q-table from {self.q_table_file}")
        except FileNotFoundError:
            self.q_table = None
            print(f"Error: Q-table file '{self.q_table_file}' not found.")

    def get_q_value(self, state, action):
        return self.q_table[state[0], state[1], action]

    def get_action(self, state):
        return int(np.argmax(self.q_table[state[0], state[1], :]))


# -------------------------
# Main simulator
# -------------------------
class Dot_Simulator:
    """Pygame-based dot mover with shared-autonomy inference, assistance, arbitration.

    This refactor keeps all behavior identical to the provided implementation:
    - Same defaults for sizes, speeds, enums
    - Same event flow, drawing, and blending behaviors
    - Same file loading and policy-color mapping
    """
    # ---- Defaults / constants ----
    DEFAULTS = {
        "GAMMA": 0.4,
        "SCREEN_WIDTH": 600,
        "SCREEN_HEIGHT": 600,
        "TOP_PANEL_HEIGHT": 175,
        "CHECKBOX_SIZE": 20,
        "DOT_RADIUS": 8,
        "DOT_SPEED": 5,
        "GRID_SIZE": 20,
        "TEXT_SIZE": 15,
        "CLICK_COOLDOWN": 50,  # ms
    }

    # action index -> (dx, dy)
    INDEX_TO_TUPLE = {
        0: (0, -1),
        1: (0, 1),
        2: (-1, 0),
        3: (1, 0),
    }

    PREDICTOR_MAP = {
        Inference.BAYESIAN: BayesianPredictor,
        Inference.MAX_ENT: MaxEntPredictor,
        Inference.CRF: CRFPredictor,
    }

    def __init__(self, policy_dir="trained_policies",
                 inference_type=Inference.BAYESIAN,
                 assistance_type=Assistance.DISTRIBUTION,
                 arbitration_type=Arbitration.LINEAR):

        # --- configuration ---
        self.GAMMA = self.DEFAULTS["GAMMA"]
        self.SCREEN_WIDTH = self.DEFAULTS["SCREEN_WIDTH"]
        self.SCREEN_HEIGHT = self.DEFAULTS["SCREEN_HEIGHT"]
        self.TOP_PANEL_HEIGHT = self.DEFAULTS["TOP_PANEL_HEIGHT"]
        self.CHECKBOX_SIZE = self.DEFAULTS["CHECKBOX_SIZE"]
        self.DOT_RADIUS = self.DEFAULTS["DOT_RADIUS"]
        self.DOT_SPEED = self.DEFAULTS["DOT_SPEED"]
        self.GRID_SIZE = self.DEFAULTS["GRID_SIZE"]
        self.TEXT_SIZE = self.DEFAULTS["TEXT_SIZE"]
        self.CLICK_COOLDOWN = self.DEFAULTS["CLICK_COOLDOWN"]

        # enums
        self.INFERENCE_TYPE = inference_type
        self.ASSISTANCE_TYPE = assistance_type
        self.ARBITRATION_TYPE = arbitration_type

        # UI toggles
        self.prob_visualization_on = True
        self.goal_visualization_on = True

        # checkbox positions (kept identical)
        self.PROB_CHECKBOX_POS = (10, self.TOP_PANEL_HEIGHT - self.CHECKBOX_SIZE - 140)
        self.GOAL_CHECKBOX_POS = (200, self.TOP_PANEL_HEIGHT - self.CHECKBOX_SIZE - 140)

        # colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)

        # state
        self.dot_x = self.SCREEN_WIDTH // 2
        self.dot_y = self.SCREEN_HEIGHT // 2
        self.ACTION_SPACE_LEN = 4
        self.last_click_time = 0

        # init pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT + self.TOP_PANEL_HEIGHT))
        caption = "Dot Mover Simulation: " + inference_type.value + ", " + assistance_type.value + ", " + arbitration_type.value
        pygame.display.set_caption(caption)
        pygame.font.init()
        
        self.font = pygame.freetype.SysFont("Arial", self.TEXT_SIZE)
        self.clock = pygame.time.Clock()

        # load policies
        self.POLICY_DIR = policy_dir
        self.POLICY_FILES = [f for f in listdir(self.POLICY_DIR) if isfile(join(self.POLICY_DIR, f))]
        self.POLICIES = [Dot_Policy(join(self.POLICY_DIR, f)) for f in self.POLICY_FILES]

        # runtime arrays
        self.prob = np.zeros(len(self.POLICIES))
        self.POLICY_COLORS = self.generate_colors(len(self.POLICIES))

        # load background images
        self.background_images = {}
        self.load_background_images()

    # -------------------------
    # background image loading
    # -------------------------
    def load_background_images(self):
        """Load background images and positions from CSV file.
        CSV format: filename,x,y, xscale, yscale
        Example: background1.png,100,200, 100,100
        """
        import csv
        
        csv_file = 'background_positions.csv'
        try:
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header row if present
                
                for row in reader:
                    if len(row) < 3:
                        continue
                    
                    filename = row[0].strip()
                    x = float(row[1])
                    y = float(row[2])
                    
                    xscale = float(row[-2])
                    yscale = float(row[-1])
                    
                    try:
                        img = pygame.image.load(os.path.join('background_images', filename)).convert_alpha()
                        # Scale if needed (adjust size as desired)
                        img = pygame.transform.scale(img, (xscale, yscale))
                        self.background_images[filename] = {'image': img, 'pos': (x, y)}
                    except FileNotFoundError:
                        print(f"Warning: background image not found: {filename}")
        except FileNotFoundError:
            print(f"Warning: {csv_file} not found. No background images will be displayed.")

    def draw_backgrounds(self):
        """Draw all background images at their specified positions."""
        for filename, data in self.background_images.items():
            img = data['image']
            x, y = data['pos']
            img_rect = img.get_rect(center=(int(x), int(y + self.TOP_PANEL_HEIGHT)))
            self.screen.blit(img, img_rect)

    # -------------------------
    # Utility & transformation
    # -------------------------
    def generate_colors(self, n=1):
        """Generate n colors that avoid pure 0 or 255 values (keeps distinct, non-black/white)."""
        colors = []
        for i in range(n):
            r = ((i + 1) * 123) % 254
            g = ((i + 1) * 231) % 254
            b = ((i + 1) * 77) % 254
            colors.append((r, g, b))
        return colors

    def get_state(self, x, y):
        """Map continuous (x,y) to a discrete grid state (sx, sy)."""
        sx = int(max(0, min(x, self.SCREEN_WIDTH - 1)) // self.GRID_SIZE)
        sy = int(max(0, min(y, self.SCREEN_HEIGHT - 1)) // self.GRID_SIZE)
        return (sx, sy)

    def index_to_tuple(self, index):
        """Return direction vector for action index. Unknown -> (0,0)."""
        return self.INDEX_TO_TUPLE.get(int(index), (0, 0))

    def execute_action(self, action):
        """Apply a unit-direction action (dx,dy) scaled by DOT_SPEED to the dot."""
        self.dot_x += self.DOT_SPEED * action[0]
        self.dot_y += self.DOT_SPEED * action[1]

    def ensure_within_boundaries(self):
        """Clamp the dot to the visible gameplay region (below top panel)."""
        self.dot_x = max(self.DOT_RADIUS, min(self.SCREEN_WIDTH - self.DOT_RADIUS, self.dot_x))
        self.dot_y = max(self.DOT_RADIUS, min(self.SCREEN_HEIGHT - self.DOT_RADIUS, self.dot_y))

    # -------------------------
    # Drawing helpers
    # -------------------------
    def draw_top_panel_background(self):
        pygame.draw.rect(self.screen, (25, 25, 25), pygame.Rect(0, 0, self.SCREEN_WIDTH, self.TOP_PANEL_HEIGHT))
        pygame.draw.line(self.screen, (180, 180, 180), (0, self.TOP_PANEL_HEIGHT - 2), (self.SCREEN_WIDTH, self.TOP_PANEL_HEIGHT - 2), 3)

    def draw_probability_bars(self):
        """Draw the horizontal probability bars in the top panel (keeps labels & geometry)."""
        num_policies = len(self.prob) or 1
        bar_width = self.SCREEN_WIDTH // num_policies
        max_bar_height = self.TOP_PANEL_HEIGHT - 75
        self.draw_top_panel_background()

        for i, p in enumerate(self.prob):
            x = i * bar_width
            bar_x = x + 8
            bar_w = bar_width - 16
            filled_h = int(max_bar_height * float(p))
            bar_bottom_y = self.TOP_PANEL_HEIGHT - 5

            # background track & filled bar
            pygame.draw.rect(self.screen, (80, 80, 80), pygame.Rect(bar_x, bar_bottom_y - max_bar_height, bar_w, max_bar_height))
            pygame.draw.rect(self.screen, self.POLICY_COLORS[i], pygame.Rect(bar_x, bar_bottom_y - filled_h, bar_w, filled_h))

            # label
            label = f"{i}: {p*100:.1f}%"
            text_surface, _ = self.font.render(label, self.WHITE)
            self.screen.blit(text_surface, (bar_x, 50))

    def draw_checkbox(self):
        """Draw both checkboxes with labels and filled rectangle when enabled."""
        def _draw_box(pos, enabled, label):
            pygame.draw.rect(self.screen, self.WHITE, pygame.Rect(pos[0], pos[1], self.CHECKBOX_SIZE, self.CHECKBOX_SIZE), 2)
            if enabled:
                # filled rectangle inside the outline
                pygame.draw.rect(self.screen, self.WHITE, pygame.Rect(pos[0], pos[1], self.CHECKBOX_SIZE, self.CHECKBOX_SIZE))
            lbl_surf, _ = self.font.render(label, self.WHITE)
            self.screen.blit(lbl_surf, (pos[0] + self.CHECKBOX_SIZE + 5, pos[1] - 2))

        _draw_box(self.GOAL_CHECKBOX_POS, self.goal_visualization_on, "Show Goals")
        _draw_box(self.PROB_CHECKBOX_POS, self.prob_visualization_on, "Show Probabilities")

    def draw_goal_visualizations(self):
        """Draw goals (circle/point) using training->visualization correspondence."""
        for i, f in enumerate(self.POLICY_FILES):
            draw_data = policy_drawing_correspondences.get_data_by_filename(f)
            if draw_data is None:
                continue
            color = self.POLICY_COLORS[i]
            x = float(draw_data["x"])
            y = float(draw_data["y"]) + self.TOP_PANEL_HEIGHT
            if draw_data["type"] == "circle":
                pygame.draw.circle(self.screen, color, (x, y), float(draw_data["r"]), width=1)
            elif draw_data["type"] == "point":
                pygame.draw.circle(self.screen, color, (x, y), self.DOT_RADIUS)

    def redraw_screen(self):
        """Clear and redraw everything (top panel + backgrounds + dot + checkboxes)."""
        # bottom gameplay region background (fill entire screen first to avoid artifacts)
        self.screen.fill(self.BLACK)

        # Draw backgrounds FIRST (so they appear behind goal visualizations and the dot)
        self.draw_backgrounds()

        if self.prob_visualization_on:
            self.draw_probability_bars()
        if self.goal_visualization_on:
            self.draw_goal_visualizations()

        # draw the dot (note: gameplay y is offset by TOP_PANEL_HEIGHT)
        dot_screen_y = self.dot_y + self.TOP_PANEL_HEIGHT
        pygame.draw.circle(self.screen, self.WHITE, (int(self.dot_x), int(dot_screen_y)), self.DOT_RADIUS)

        self.draw_checkbox()
        pygame.display.flip()
        self.clock.tick(60)

    # -------------------------
    # Core loop and helpers
    # -------------------------
    def _create_predictor(self):
        cls = self.PREDICTOR_MAP.get(self.INFERENCE_TYPE, BayesianPredictor)
        return cls(self.POLICIES)

    def _create_assistant(self):
        # if self.ASSISTANCE_TYPE is Assistance.DISTRIBUTION:
        #     return SharedAutoPolicy(self.POLICIES, list(range(self.ACTION_SPACE_LEN)))
        # return None
        return SharedAutoPolicy(self.POLICIES, list(range(self.ACTION_SPACE_LEN)))

    def _handle_mouse_clicks(self, event):
        """Flip checkboxes if clicked, respecting cooldown (keeps same logic as original)."""
        now = pygame.time.get_ticks()
        if now - self.last_click_time < self.CLICK_COOLDOWN:
            return
        self.last_click_time = now

        mouse_pos = event.pos
        prob_rect = pygame.Rect(self.PROB_CHECKBOX_POS, (self.CHECKBOX_SIZE, self.CHECKBOX_SIZE))
        goal_rect = pygame.Rect(self.GOAL_CHECKBOX_POS, (self.CHECKBOX_SIZE, self.CHECKBOX_SIZE))

        if prob_rect.collidepoint(mouse_pos):
            self.prob_visualization_on = not self.prob_visualization_on
        if goal_rect.collidepoint(mouse_pos):
            self.goal_visualization_on = not self.goal_visualization_on

    def blend(self, u, a):
        """
        Combine user command u (index) and robot action a (index) according to arbitration:
        - LINEAR: linear blend by GAMMA (then normalize)
        - PROBABILISTIC: weight by robot confidence (max of self.prob)
        - ONLY_USER: return user vector
        """
        u_vec = np.array(self.index_to_tuple(u), dtype=float)
        a_vec = np.array(self.index_to_tuple(a), dtype=float)

        if self.ARBITRATION_TYPE is Arbitration.ONLY_USER:
            return tuple(u_vec.tolist())

        if self.ARBITRATION_TYPE is Arbitration.LINEAR:
            blended = (u_vec * self.GAMMA) + (a_vec * (1 - self.GAMMA))
        elif self.ARBITRATION_TYPE is Arbitration.PROBABILISTIC:
            self.robot_confidence = float(max(self.prob)) if len(self.prob) else 0.0
            p_robot = self.robot_confidence
            blended = (p_robot * a_vec) + ((1 - p_robot) * u_vec)
        else:
            blended = u_vec  # fallback

        mag = np.linalg.norm(blended)
        if mag == 0:
            return (0, 0)
        return (blended[0] / mag, blended[1] / mag)

    def run_shared(self):
        """Main loop: handle input, inference, assistance, arbitration, drawing."""
        pred = self._create_predictor()
        policy = self._create_assistant()

        running = True

        while running:
            # --- events ---
            last_event = None
            for event in pygame.event.get():
                last_event = event
                if event.type == pygame.QUIT:
                    running = False

            # --- Check user action for THIS frame ---
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                u = 0
            elif keys[pygame.K_DOWN]:
                u = 1
            elif keys[pygame.K_LEFT]:
                u = 2
            elif keys[pygame.K_RIGHT]:
                u = 3
            else:
                u = -1   # no user input this frame

            # --- If no user action: robot does nothing ---
            if u == -1:
                self.redraw_screen()

                # still allow checkbox clicks
                if last_event and last_event.type == pygame.MOUSEBUTTONDOWN and last_event.button == 1:
                    self._handle_mouse_clicks(last_event)

                continue

            # --- Inference
            self.prob = pred.update(self.get_state(self.dot_x, self.dot_y), u)

            # --- Assistance
            optimal_action = policy.get_action(
                self.get_state(self.dot_x, self.dot_y),
                self.prob
            )

            # --- Arbitration & Execution
            blended_action = self.blend(u, optimal_action)
            self.execute_action(blended_action)

            # --- boundaries & draw ---
            self.ensure_within_boundaries()
            self.redraw_screen()

            # --- handle UI clicks ---
            if last_event and last_event.type == pygame.MOUSEBUTTONDOWN and last_event.button == 1:
                self._handle_mouse_clicks(last_event)


# -------------------------
# CLI-like selection & launch (kept identical)
# -------------------------
if __name__ == "__main__": 
    # inference selection
    inference_selector = Method_Selector(options=[i for i in Inference], caption="Inference Method")
    inference_type = inference_selector.get()

    # arbitration selection
    arbitration_selector = Method_Selector(options=[a for a in Arbitration], caption="Arbitration Method")
    arbitration_type = arbitration_selector.get()

    # instantiate & run (assistance left to default as in original)
    dot = Dot_Simulator(inference_type=inference_type, arbitration_type=arbitration_type)
    dot.run_shared()
# # Refactored Dot Simulator (behavior preserved)
# import sys
# import os
# from os import listdir
# from os.path import isfile, join

# import numpy as np
# import pygame
# import pygame.freetype

# from util.shared_auto import SharedAutoPolicy
# from util.predictors import BayesianPredictor, MaxEntPredictor, CRFPredictor
# from util.selector import Method_Selector
# from util.SA_types import Inference, Assistance, Arbitration
# from training import policy_drawing_correspondences


# # -------------------------
# # Lightweight Q-table wrapper
# # -------------------------
# class Dot_Policy:
#     """Simple adapter around a saved Q-table .npy file.

#     Public API:
#         get_q_value(state, action)
#         get_action(state) -> argmax over actions
#     """
#     def __init__(self, q_table_file="q_table_topleft.npy"):
#         self.q_table_file = q_table_file
#         try:
#             self.q_table = np.load(self.q_table_file)
#             print(f"Loaded Q-table from {self.q_table_file}")
#         except FileNotFoundError:
#             self.q_table = None
#             print(f"Error: Q-table file '{self.q_table_file}' not found.")

#     def get_q_value(self, state, action):
#         return self.q_table[state[0], state[1], action]

#     def get_action(self, state):
#         return int(np.argmax(self.q_table[state[0], state[1], :]))


# # -------------------------
# # Main simulator
# # -------------------------
# class Dot_Simulator:
#     """Pygame-based dot mover with shared-autonomy inference, assistance, arbitration.

#     This refactor keeps all behavior identical to the provided implementation:
#     - Same defaults for sizes, speeds, enums
#     - Same event flow, drawing, and blending behaviors
#     - Same file loading and policy-color mapping
#     """
#     # ---- Defaults / constants ----
#     DEFAULTS = {
#         "GAMMA": 0.4,
#         "SCREEN_WIDTH": 600,
#         "SCREEN_HEIGHT": 600,
#         "TOP_PANEL_HEIGHT": 175,
#         "CHECKBOX_SIZE": 20,
#         "DOT_RADIUS": 8,
#         "DOT_SPEED": 5,
#         "GRID_SIZE": 20,
#         "TEXT_SIZE": 15,
#         "CLICK_COOLDOWN": 50,  # ms
#     }

#     # action index -> (dx, dy)
#     INDEX_TO_TUPLE = {
#         0: (0, -1),
#         1: (0, 1),
#         2: (-1, 0),
#         3: (1, 0),
#     }

#     PREDICTOR_MAP = {
#         Inference.BAYESIAN: BayesianPredictor,
#         Inference.MAX_ENT: MaxEntPredictor,
#         Inference.CRF: CRFPredictor,
#     }

#     def __init__(self, policy_dir="trained_policies",
#                  inference_type=Inference.BAYESIAN,
#                  assistance_type=Assistance.DISTRIBUTION,
#                  arbitration_type=Arbitration.LINEAR):

#         # --- configuration ---
#         self.GAMMA = self.DEFAULTS["GAMMA"]
#         self.SCREEN_WIDTH = self.DEFAULTS["SCREEN_WIDTH"]
#         self.SCREEN_HEIGHT = self.DEFAULTS["SCREEN_HEIGHT"]
#         self.TOP_PANEL_HEIGHT = self.DEFAULTS["TOP_PANEL_HEIGHT"]
#         self.CHECKBOX_SIZE = self.DEFAULTS["CHECKBOX_SIZE"]
#         self.DOT_RADIUS = self.DEFAULTS["DOT_RADIUS"]
#         self.DOT_SPEED = self.DEFAULTS["DOT_SPEED"]
#         self.GRID_SIZE = self.DEFAULTS["GRID_SIZE"]
#         self.TEXT_SIZE = self.DEFAULTS["TEXT_SIZE"]
#         self.CLICK_COOLDOWN = self.DEFAULTS["CLICK_COOLDOWN"]

#         # enums
#         self.INFERENCE_TYPE = inference_type
#         self.ASSISTANCE_TYPE = assistance_type
#         self.ARBITRATION_TYPE = arbitration_type

#         # UI toggles
#         self.prob_visualization_on = True
#         self.goal_visualization_on = True

#         # checkbox positions (kept identical)
#         self.PROB_CHECKBOX_POS = (10, self.TOP_PANEL_HEIGHT - self.CHECKBOX_SIZE - 140)
#         self.GOAL_CHECKBOX_POS = (200, self.TOP_PANEL_HEIGHT - self.CHECKBOX_SIZE - 140)

#         # colors
#         self.BLACK = (0, 0, 0)
#         self.WHITE = (255, 255, 255)

#         # state
#         self.dot_x = self.SCREEN_WIDTH // 2
#         self.dot_y = self.SCREEN_HEIGHT // 2
#         self.ACTION_SPACE_LEN = 4
#         self.last_click_time = 0

#         # init pygame
#         pygame.init()
#         self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT + self.TOP_PANEL_HEIGHT))
#         caption = "Dot Mover Simulation: " + inference_type.value + ", " + assistance_type.value + ", " + arbitration_type.value
#         pygame.display.set_caption(caption)
#         pygame.font.init()
        
#         self.font = pygame.freetype.SysFont("Arial", self.TEXT_SIZE)
#         self.clock = pygame.time.Clock()

#         # load policies
#         self.POLICY_DIR = policy_dir
#         self.POLICY_FILES = [f for f in listdir(self.POLICY_DIR) if isfile(join(self.POLICY_DIR, f))]
#         self.POLICIES = [Dot_Policy(join(self.POLICY_DIR, f)) for f in self.POLICY_FILES]

#         # runtime arrays
#         self.prob = np.zeros(len(self.POLICIES))
#         self.POLICY_COLORS = self.generate_colors(len(self.POLICIES))

#     # -------------------------
#     # Utility & transformation
#     # -------------------------
#     def generate_colors(self, n=1):
#         """Generate n colors that avoid pure 0 or 255 values (keeps distinct, non-black/white)."""
#         colors = []
#         for i in range(n):
#             r = ((i + 1) * 123) % 254
#             g = ((i + 1) * 231) % 254
#             b = ((i + 1) * 77) % 254
#             colors.append((r, g, b))
#         return colors

#     def get_state(self, x, y):
#         """Map continuous (x,y) to a discrete grid state (sx, sy)."""
#         sx = int(max(0, min(x, self.SCREEN_WIDTH - 1)) // self.GRID_SIZE)
#         sy = int(max(0, min(y, self.SCREEN_HEIGHT - 1)) // self.GRID_SIZE)
#         return (sx, sy)

#     def index_to_tuple(self, index):
#         """Return direction vector for action index. Unknown -> (0,0)."""
#         return self.INDEX_TO_TUPLE.get(int(index), (0, 0))

#     def execute_action(self, action):
#         """Apply a unit-direction action (dx,dy) scaled by DOT_SPEED to the dot."""
#         self.dot_x += self.DOT_SPEED * action[0]
#         self.dot_y += self.DOT_SPEED * action[1]

#     def ensure_within_boundaries(self):
#         """Clamp the dot to the visible gameplay region (below top panel)."""
#         self.dot_x = max(self.DOT_RADIUS, min(self.SCREEN_WIDTH - self.DOT_RADIUS, self.dot_x))
#         self.dot_y = max(self.DOT_RADIUS, min(self.SCREEN_HEIGHT - self.DOT_RADIUS, self.dot_y))

#     # -------------------------
#     # Drawing helpers
#     # -------------------------
#     def draw_top_panel_background(self):
#         pygame.draw.rect(self.screen, (25, 25, 25), pygame.Rect(0, 0, self.SCREEN_WIDTH, self.TOP_PANEL_HEIGHT))
#         pygame.draw.line(self.screen, (180, 180, 180), (0, self.TOP_PANEL_HEIGHT - 2), (self.SCREEN_WIDTH, self.TOP_PANEL_HEIGHT - 2), 3)

#     def draw_probability_bars(self):
#         """Draw the horizontal probability bars in the top panel (keeps labels & geometry)."""
#         num_policies = len(self.prob) or 1
#         bar_width = self.SCREEN_WIDTH // num_policies
#         max_bar_height = self.TOP_PANEL_HEIGHT - 75
#         self.draw_top_panel_background()

#         for i, p in enumerate(self.prob):
#             x = i * bar_width
#             bar_x = x + 8
#             bar_w = bar_width - 16
#             filled_h = int(max_bar_height * float(p))
#             bar_bottom_y = self.TOP_PANEL_HEIGHT - 5

#             # background track & filled bar
#             pygame.draw.rect(self.screen, (80, 80, 80), pygame.Rect(bar_x, bar_bottom_y - max_bar_height, bar_w, max_bar_height))
#             pygame.draw.rect(self.screen, self.POLICY_COLORS[i], pygame.Rect(bar_x, bar_bottom_y - filled_h, bar_w, filled_h))

#             # label
#             label = f"{i}: {p*100:.1f}%"
#             text_surface, _ = self.font.render(label, self.WHITE)
#             self.screen.blit(text_surface, (bar_x, 50))

#     def draw_checkbox(self):
#         """Draw both checkboxes with labels and filled rectangle when enabled."""
#         def _draw_box(pos, enabled, label):
#             pygame.draw.rect(self.screen, self.WHITE, pygame.Rect(pos[0], pos[1], self.CHECKBOX_SIZE, self.CHECKBOX_SIZE), 2)
#             if enabled:
#                 # filled rectangle inside the outline
#                 pygame.draw.rect(self.screen, self.WHITE, pygame.Rect(pos[0], pos[1], self.CHECKBOX_SIZE, self.CHECKBOX_SIZE))
#             lbl_surf, _ = self.font.render(label, self.WHITE)
#             self.screen.blit(lbl_surf, (pos[0] + self.CHECKBOX_SIZE + 5, pos[1] - 2))

#         _draw_box(self.GOAL_CHECKBOX_POS, self.goal_visualization_on, "Show Goals")
#         _draw_box(self.PROB_CHECKBOX_POS, self.prob_visualization_on, "Show Probabilities")

#     def draw_goal_visualizations(self):
#         """Draw goals (circle/point) using training->visualization correspondence."""
#         for i, f in enumerate(self.POLICY_FILES):
#             draw_data = policy_drawing_correspondences.get_data_by_filename(f)
#             if draw_data is None:
#                 continue
#             color = self.POLICY_COLORS[i]
#             x = float(draw_data["x"])
#             y = float(draw_data["y"]) + self.TOP_PANEL_HEIGHT
#             if draw_data["type"] == "circle":
#                 pygame.draw.circle(self.screen, color, (x, y), float(draw_data["r"]), width=1)
#             elif draw_data["type"] == "point":
#                 pygame.draw.circle(self.screen, color, (x, y), self.DOT_RADIUS)

#     def redraw_screen(self):
#         """Clear and redraw everything (top panel + dot + checkboxes)."""
#         # bottom gameplay region background (fill entire screen first to avoid artifacts)
#         self.screen.fill(self.BLACK)

#         if self.prob_visualization_on:
#             self.draw_probability_bars()
#         if self.goal_visualization_on:
#             self.draw_goal_visualizations()

#         # draw the dot (note: gameplay y is offset by TOP_PANEL_HEIGHT)
#         dot_screen_y = self.dot_y + self.TOP_PANEL_HEIGHT
#         pygame.draw.circle(self.screen, self.WHITE, (int(self.dot_x), int(dot_screen_y)), self.DOT_RADIUS)

#         self.draw_checkbox()
#         pygame.display.flip()
#         self.clock.tick(60)

#     # -------------------------
#     # Core loop and helpers
#     # -------------------------
#     def _create_predictor(self):
#         cls = self.PREDICTOR_MAP.get(self.INFERENCE_TYPE, BayesianPredictor)
#         return cls(self.POLICIES)

#     def _create_assistant(self):
#         # if self.ASSISTANCE_TYPE is Assistance.DISTRIBUTION:
#         #     return SharedAutoPolicy(self.POLICIES, list(range(self.ACTION_SPACE_LEN)))
#         # return None
#         return SharedAutoPolicy(self.POLICIES, list(range(self.ACTION_SPACE_LEN)))

#     def _handle_mouse_clicks(self, event):
#         """Flip checkboxes if clicked, respecting cooldown (keeps same logic as original)."""
#         now = pygame.time.get_ticks()
#         if now - self.last_click_time < self.CLICK_COOLDOWN:
#             return
#         self.last_click_time = now

#         mouse_pos = event.pos
#         prob_rect = pygame.Rect(self.PROB_CHECKBOX_POS, (self.CHECKBOX_SIZE, self.CHECKBOX_SIZE))
#         goal_rect = pygame.Rect(self.GOAL_CHECKBOX_POS, (self.CHECKBOX_SIZE, self.CHECKBOX_SIZE))

#         if prob_rect.collidepoint(mouse_pos):
#             self.prob_visualization_on = not self.prob_visualization_on
#         if goal_rect.collidepoint(mouse_pos):
#             self.goal_visualization_on = not self.goal_visualization_on

#     def blend(self, u, a):
#         """
#         Combine user command u (index) and robot action a (index) according to arbitration:
#         - LINEAR: linear blend by GAMMA (then normalize)
#         - PROBABILISTIC: weight by robot confidence (max of self.prob)
#         - ONLY_USER: return user vector
#         """
#         u_vec = np.array(self.index_to_tuple(u), dtype=float)
#         a_vec = np.array(self.index_to_tuple(a), dtype=float)

#         if self.ARBITRATION_TYPE is Arbitration.ONLY_USER:
#             return tuple(u_vec.tolist())

#         if self.ARBITRATION_TYPE is Arbitration.LINEAR:
#             blended = (u_vec * self.GAMMA) + (a_vec * (1 - self.GAMMA))
#         elif self.ARBITRATION_TYPE is Arbitration.PROBABILISTIC:
#             self.robot_confidence = float(max(self.prob)) if len(self.prob) else 0.0
#             p_robot = self.robot_confidence
#             blended = (p_robot * a_vec) + ((1 - p_robot) * u_vec)
#         else:
#             blended = u_vec  # fallback

#         mag = np.linalg.norm(blended)
#         if mag == 0:
#             return (0, 0)
#         return (blended[0] / mag, blended[1] / mag)

#     def run_shared(self):
#         """Main loop: handle input, inference, assistance, arbitration, drawing."""
#         pred = self._create_predictor()
#         policy = self._create_assistant()

#         running = True

#         while running:
#             # --- events ---
#             last_event = None
#             for event in pygame.event.get():
#                 last_event = event
#                 if event.type == pygame.QUIT:
#                     running = False

#             # --- Check user action for THIS frame ---
#             keys = pygame.key.get_pressed()
#             if keys[pygame.K_UP]:
#                 u = 0
#             elif keys[pygame.K_DOWN]:
#                 u = 1
#             elif keys[pygame.K_LEFT]:
#                 u = 2
#             elif keys[pygame.K_RIGHT]:
#                 u = 3
#             else:
#                 u = -1   # no user input this frame

#             # --- If no user action: robot does nothing ---
#             if u == -1:
#                 self.redraw_screen()

#                 # still allow checkbox clicks
#                 if last_event and last_event.type == pygame.MOUSEBUTTONDOWN and last_event.button == 1:
#                     self._handle_mouse_clicks(last_event)

#                 continue

#             # --- Inference
#             self.prob = pred.update(self.get_state(self.dot_x, self.dot_y), u)

#             # --- Assistance
#             optimal_action = policy.get_action(
#                 self.get_state(self.dot_x, self.dot_y),
#                 self.prob
#             )

#             # --- Arbitration & Execution
#             blended_action = self.blend(u, optimal_action)
#             self.execute_action(blended_action)

#             # --- boundaries & draw ---
#             self.ensure_within_boundaries()
#             self.redraw_screen()

#             # --- handle UI clicks ---
#             if last_event and last_event.type == pygame.MOUSEBUTTONDOWN and last_event.button == 1:
#                 self._handle_mouse_clicks(last_event)


# # -------------------------
# # CLI-like selection & launch (kept identical)
# # -------------------------
# if __name__ == "__main__": 
#     # inference selection
#     inference_selector = Method_Selector(options=[i for i in Inference], caption="Inference Method")
#     inference_type = inference_selector.get()

#     # arbitration selection
#     arbitration_selector = Method_Selector(options=[a for a in Arbitration], caption="Arbitration Method")
#     arbitration_type = arbitration_selector.get()

#     # instantiate & run (assistance left to default as in original)
#     dot = Dot_Simulator(inference_type=inference_type, arbitration_type=arbitration_type)
#     dot.run_shared()
