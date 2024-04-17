import pygame
import numpy as np
from pygame.sprite import Sprite

class Agent(Sprite):
    def __init__(self, screen, base_image):
        Sprite.__init__(self)
        
        self.screen = screen
        self.base_image = base_image
        self.ready_to_draw = False

    def update(self, position, rotation):
        def to_degrees(r):
            return r * 180 / np.pi

        # Rotation
        self.image = pygame.transform.rotate(self.base_image, -to_degrees(rotation))
        self.image_w, self.image_h = self.image.get_size()

        # Position
        self.position = position
        
        self.ready_to_draw = True

    def draw(self):
        assert self.ready_to_draw

        self.draw_rect = self.image.get_rect().move(
                self.position[0]- self.image_w / 2, 
                self.position[1] - self.image_h / 2)
        self.screen.blit(self.image, self.draw_rect)

        self.ready_to_draw = False
        

class Visualizer:
    WINDOW_W, WINDOW_H = 500, 500
    VEH_WIDTH, VEH_HEIGHT = 2.05, 4.6
    GOAL_RADIUS = 2
    COLOR_LIST = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 165, 0),  # Orange
    ]
    BACKGROUND_COLOR = (0,0,0)
    NO_ROTATION = 0

    def __init__(self, agent_count, human_render=True):
        self.human_render = human_render
        
        pygame.init()
        pygame.font.init()
        pygame.display.init() # TODO(sk): unnecessary?
        
        # TODO(sk): check depth parameter
        self.screen = pygame.display.set_mode((self.WINDOW_W, self.WINDOW_H), 0, 32)

        self.clock = pygame.time.Clock()

        self.agent_image = pygame.image.load("/home/samk/gpudrive/agent.png").convert_alpha()

        self.agents = pygame.sprite.Group()
        for i in range(agent_count):
            self.agents.add(Agent(self.screen, self.agent_image))

    def scale_coords(self, coords, x_avg, y_avg):
        """Scale the coordinates to fit within the pygame surface window and center them.
        Args:
            coords: x, y coordinates
        """
        x, y = coords
        x_scaled = x - x_avg + (self.WINDOW_W / 2)
        y_scaled = y - y_avg + (self.WINDOW_H / 2)
        return (x_scaled, y_scaled)

    def _create_image_array(self, surf):
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)
        )

    def draw(self, positions, rotations, goals, mask):
        self.screen.fill(self.BACKGROUND_COLOR)
         
        agent_pos_mean = np.mean(positions, axis=0, where=mask == 1)
        
        for agent_idx, agent in enumerate(self.agents):
            if not mask[agent_idx]:
                continue

            # Agent
            current_pos_scaled = self.scale_coords(
                positions[agent_idx], 
                agent_pos_mean[0], 
                agent_pos_mean[1]
            )
            agent.update(current_pos_scaled, rotations[agent_idx])
            agent.draw()

            # Goal
            current_goal_scaled = self.scale_coords(
                goals[agent_idx],
                agent_pos_mean[0],
                agent_pos_mean[1]
            )
            pygame.draw.circle(
                surface=self.screen,
                color=(255, 0, 0),
                center=(
                    int(current_goal_scaled[0]),
                    int(current_goal_scaled[1]),
                ),
                radius=self.GOAL_RADIUS,
            )

        if self.human_render:
            self.clock.tick(30) # Limit to 30 FPS
            pygame.display.flip()
        else:
            return self._create_image_array(self.screen)

    def destroy(self):
        pygame.display.quit()
        pygame.quit()

