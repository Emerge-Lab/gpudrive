import pygame
from pygame import Color
import numpy as np
from pygame.sprite import Sprite
import os


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
            self.position[0] - self.image_w / 2, self.position[1] - self.image_h / 2
        )
        self.screen.blit(self.image, self.draw_rect)

        self.ready_to_draw = False


class Visualizer:
    WINDOW_W, WINDOW_H = 500, 500
    VEH_WIDTH, VEH_HEIGHT = 2.05, 4.6
    GOAL_RADIUS = 2
    BACKGROUND_COLOR = (0, 0, 0)

    def __init__(self, agent_count, human_render=True):
        self.human_render = human_render

        pygame.init()
        pygame.font.init()
        pygame.display.init()  # TODO(sk): unnecessary?

        # TODO(sk): check depth parameter
        self.screen = pygame.display.set_mode((self.WINDOW_W, self.WINDOW_H), 0, 32)

        self.clock = pygame.time.Clock()

        self.agent_images = []
        for base_name in ["green_agent.svg", "pink_agent.svg", "yellow_agent.svg"]:
            absolute_image_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", base_name
            )
            self.agent_images.append(
                pygame.image.load(absolute_image_path).convert_alpha()
            )

        self.agents = pygame.sprite.Group()
        for i in range(agent_count):
            self.agents.add(
                Agent(self.screen, self.agent_images[i % len(self.agent_images)])
            )

        self.colors = [Color("green"), Color("pink"), Color("yellow")]

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
        return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))

    def draw(self, positions, rotations, goals, mask):
        self.screen.fill(self.BACKGROUND_COLOR)

        # agent_pos_mean = np.mean(positions, axis=0, where=np.expand_dims(mask, axis=1))
        agent_pos_mean = positions[0]

        for agent_idx, agent in enumerate(self.agents):
            if not mask[agent_idx]:
                continue

            # Agent
            current_pos_scaled = self.scale_coords(
                positions[agent_idx], agent_pos_mean[0], agent_pos_mean[1]
            )
            agent.update(current_pos_scaled, rotations[agent_idx])
            agent.draw()

            # Goal
            current_goal_scaled = self.scale_coords(
                goals[agent_idx], agent_pos_mean[0], agent_pos_mean[1]
            )
            pygame.draw.circle(
                surface=self.screen,
                color=self.colors[agent_idx % len(self.colors)],
                center=(
                    int(current_goal_scaled[0]),
                    int(current_goal_scaled[1]),
                ),
                radius=self.GOAL_RADIUS,
            )

        if self.human_render:
            self.clock.tick(30)  # Limit to 30 FPS
            pygame.display.flip()
        else:
            return self._create_image_array(self.screen)

    def destroy(self):
        pygame.display.quit()
        pygame.quit()
