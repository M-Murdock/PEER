"""
Example Pygame Simulator for Web
Save this as: main.py

To convert to web:
1. pip install pygbag
2. pygbag main.py
3. Copy the build/web folder to your study website
"""

import pygame
import asyncio
import sys

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Study Simulator - Click the Target")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)

# Game variables
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)

# Target
target_pos = [WIDTH // 2, HEIGHT // 2]
target_radius = 40
score = 0
clicks = 0
task_complete = False

def draw_target(pos, radius):
    """Draw the target circle"""
    pygame.draw.circle(screen, RED, pos, radius)
    pygame.draw.circle(screen, WHITE, pos, radius - 10)
    pygame.draw.circle(screen, RED, pos, radius - 20)

def check_target_click(mouse_pos, target_pos, radius):
    """Check if mouse click hit the target"""
    dx = mouse_pos[0] - target_pos[0]
    dy = mouse_pos[1] - target_pos[1]
    distance = (dx**2 + dy**2)**0.5
    return distance <= radius

def move_target():
    """Move target to random position"""
    import random
    margin = 100
    target_pos[0] = random.randint(margin, WIDTH - margin)
    target_pos[1] = random.randint(margin, HEIGHT - margin)

def send_results_to_html():
    """Send results to parent HTML page (only works in browser)"""
    try:
        # Check if running in browser
        if sys.platform == "emscripten":
            # This will work when deployed with Pygbag
            from browser import window
            results = {
                "type": "simulator_complete",
                "score": score,
                "clicks": clicks,
                "accuracy": score / clicks if clicks > 0 else 0
            }
            window.parent.postMessage(results, "*")
    except:
        # Running locally - just print
        print(f"Results: Score={score}, Clicks={clicks}")

async def main():
    """Main game loop - MUST be async for web compatibility"""
    global score, clicks, task_complete
    
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN and not task_complete:
                clicks += 1
                mouse_pos = pygame.mouse.get_pos()
                
                if check_target_click(mouse_pos, target_pos, target_radius):
                    score += 1
                    move_target()
                    
                    # Complete task after 10 successful hits
                    if score >= 10:
                        task_complete = True
                        send_results_to_html()
        
        # Drawing
        screen.fill(WHITE)
        
        if not task_complete:
            # Draw target
            draw_target(target_pos, target_radius)
            
            # Draw instructions
            instruction = small_font.render("Click the target 10 times", True, BLACK)
            screen.blit(instruction, (WIDTH // 2 - instruction.get_width() // 2, 30))
            
            # Draw score
            score_text = font.render(f"Score: {score}/10", True, BLACK)
            screen.blit(score_text, (20, 20))
            
            # Draw clicks
            clicks_text = small_font.render(f"Total Clicks: {clicks}", True, BLACK)
            screen.blit(clicks_text, (20, 60))
        else:
            # Task complete screen
            complete_text = font.render("Task Complete!", True, GREEN)
            screen.blit(complete_text, (WIDTH // 2 - complete_text.get_width() // 2, HEIGHT // 2 - 60))
            
            accuracy = (score / clicks * 100) if clicks > 0 else 0
            accuracy_text = small_font.render(f"Accuracy: {accuracy:.1f}%", True, BLACK)
            screen.blit(accuracy_text, (WIDTH // 2 - accuracy_text.get_width() // 2, HEIGHT // 2 - 10))
            
            continue_text = small_font.render("Click 'Continue' below to proceed", True, BLUE)
            screen.blit(continue_text, (WIDTH // 2 - continue_text.get_width() // 2, HEIGHT // 2 + 30))
        
        # Update display
        pygame.display.flip()
        clock.tick(60)
        
        # CRITICAL: This allows browser to update
        await asyncio.sleep(0)
    
    pygame.quit()

# Run the game
if __name__ == "__main__":
    asyncio.run(main())
