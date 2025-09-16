import pygame
import random
import time

# Initialize Pygame
pygame.init()

# Screen setup
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Smart Bot Walker")

# Colors
WHITE = (255, 255, 255)
BLUE = (50, 100, 255)

# Bot setup
bot_size = 20
bot_x, bot_y = WIDTH // 2, HEIGHT // 2
bot_speed = 3

# Start timer
start_time = time.time()
move_enabled = False

# Game loop
running = True
clock = pygame.time.Clock()

while running:
    screen.fill(WHITE)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Enable walking after 10 sec
    if not move_enabled and time.time() - start_time >= 10:
        move_enabled = True

    # Smart random walking
    if move_enabled:
        # Pick a random direction every few frames
        if random.randint(1, 20) == 1:
            dx = random.choice([-bot_speed, 0, bot_speed])
            dy = random.choice([-bot_speed, 0, bot_speed])
        else:
            dx, dy = dx, dy  # keep going same way

        # Update bot position
        bot_x += dx
        bot_y += dy

        # Keep inside screen
        if bot_x < 0: bot_x = 0
        if bot_x > WIDTH - bot_size: bot_x = WIDTH - bot_size
        if bot_y < 0: bot_y = 0
        if bot_y > HEIGHT - bot_size: bot_y = HEIGHT - bot_size

    # Draw bot
    pygame.draw.rect(screen, BLUE, (bot_x, bot_y, bot_size, bot_size))

    pygame.display.flip()
    clock.tick(30)  # 30 FPS

pygame.quit()
