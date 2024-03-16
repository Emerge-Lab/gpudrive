import wandb
import pygame
import numpy as np
from PIL import Image

# Initialize wandb
wandb.init(project='gpudrive', group='render')

# Initialize Pygame
pygame.init()

def create_rgb_array(x_position):
    WINDOW_W, WINDOW_H = 800, 600
    surf = pygame.Surface((WINDOW_W, WINDOW_H))
    surf.fill((255, 255, 255))  # Fill with white
    # Correctly form the Rect object with (x, y, width, height)
    pygame.draw.rect(surf, (255, 0, 0), pygame.Rect(x_position, 50, 100, 50))  # Draw a red rectangle
    return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))

# Generate images for each timestep
images = []
for i in range(5):
    x_position = 50 + i*20  # Move the box to the right by 20 pixels each timestep
    rgb_array = create_rgb_array(x_position)
    images.append(rgb_array.T)

# Log the series of images as a video to wandb
wandb.log({"Moving Box": wandb.Video(np.array(images), fps=4, format="gif")})
