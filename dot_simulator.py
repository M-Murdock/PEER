# Import the pygame library
import pygame
import sys

# --- Constants ---
# Screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

# Colors (R, G, B)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Dot properties
DOT_RADIUS = 8
DOT_SPEED = 5 # Pixels to move per key press

# --- Initialization ---
# Initialize all imported pygame modules
pygame.init()

# Set up the display window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Dot Mover Simulation")

# --- Game Variables ---
# Initial position of the dot (center of the screen)
dot_x = SCREEN_WIDTH // 2
dot_y = SCREEN_HEIGHT // 2

# Clock for controlling the frame rate (though not strictly needed for this event-based movement)
clock = pygame.time.Clock()

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
    dot_x = max(DOT_RADIUS, min(SCREEN_WIDTH - DOT_RADIUS, dot_x))
    dot_y = max(DOT_RADIUS, min(SCREEN_HEIGHT - DOT_RADIUS, dot_y))

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

# --- Shutdown ---
# Once the loop is exited, quit pygame and the program
pygame.quit()
sys.exit()