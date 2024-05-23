import pygame
from pygame import Color
import numpy as np
from pygame.sprite import Sprite
import os
import math
import gpudrive


class PyGameVisualizer:
    WINDOW_W, WINDOW_H = 1920, 1080
    BACKGROUND_COLOR = (255, 255, 255)
    PADDING_PCT = 0.0
    COLOR_LIST = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 165, 0),  # Orange
    ]
    color_dict = {
        float(gpudrive.EntityType.RoadEdge): (0, 0, 0),  # Black
        float(gpudrive.EntityType.RoadLane): (225, 225, 225),  # Grey
        float(gpudrive.EntityType.RoadLine): (225, 255, 225),  # Green
        float(gpudrive.EntityType.SpeedBump): (255, 0, 255),  # Red
        float(gpudrive.EntityType.CrossWalk): (213, 20, 20),  # dark
        float(gpudrive.EntityType.StopSign): (255, 0, 255),  # Blue
    }

    def __init__(self, sim, world_render_idx, render_mode, goal_radius):
        self.sim = sim
        self.world_render_idx = world_render_idx
        self.render_mode = render_mode
        self.goal_radius = goal_radius

        self.padding_x = self.PADDING_PCT * self.WINDOW_W
        self.padding_y = self.PADDING_PCT * self.WINDOW_H

        pygame.init()
        pygame.font.init()
        self.screen = None
        self.clock = None
        if self.screen is None and self.render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.WINDOW_W, self.WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.WINDOW_W, self.WINDOW_H))
        self.compute_window_settings()
        self.init_map()

    @staticmethod
    def get_all_endpoints(map_info):
        centers = map_info[:, :2]
        lengths = map_info[:, 2]
        yaws = map_info[:, 5]

        offsets = np.column_stack((lengths * np.cos(yaws), lengths * np.sin(yaws)))
        starts = centers - offsets
        ends = centers + offsets
        return starts, ends
    
    def compute_window_settings(self):
        map_info = (
            self.sim.map_observation_tensor()
            .to_torch()[self.world_render_idx]
            .cpu()
            .numpy()
        )
        map_info = map_info[map_info[:, -1] != float(gpudrive.EntityType.Padding)]
        roads = map_info[map_info[:, -1] <= float(gpudrive.EntityType.RoadLane)]
        endpoints = PyGameVisualizer.get_all_endpoints(roads)

        all_endpoints = np.concatenate(endpoints, axis=0)
        
        # Adjust window dimensions by subtracting padding
        adjusted_window_width = self.WINDOW_W - self.padding_x
        adjusted_window_height = self.WINDOW_H - self.padding_y

        self.zoom_scale_x = adjusted_window_width / (
            all_endpoints[:, 0].max() - all_endpoints[:, 0].min() 
        ) 
        self.zoom_scale_y = adjusted_window_height / (
            all_endpoints[:, 1].max() - all_endpoints[:, 1].min()
        ) 

        # self.window_center = np.mean(all_endpoints[:, :2], axis=0)
        self.window_center = np.array([(all_endpoints[:, 0].max() + all_endpoints[:, 0].min()) / 2,
                                       (all_endpoints[:, 1].max() + all_endpoints[:, 1].min()) / 2])
        print(f"Window center: {self.window_center}")

    def create_render_mask(self):
        agent_to_is_valid = (
            self.sim.valid_state_tensor()
            .to_torch()[self.world_render_idx, :, :]
            .cpu()
            .detach()
            .numpy()
        )

        return agent_to_is_valid.astype(bool)

    def scale_coords(self, coords):
        """Scale the coordinates to fit within the pygame surface window and center them.
        Args:
            coords: x, y coordinates
        """
        x, y = coords
        x_scaled = (
            (x - self.window_center[0]) * self.zoom_scale_x
            + self.WINDOW_W / 2 - self.padding_x / 2
        )
        y_scaled = (
            (y - self.window_center[1]) * self.zoom_scale_y
            + self.WINDOW_H / 2 - self.padding_y / 2
        )

        return (x_scaled, y_scaled)

    @staticmethod
    def compute_agent_corners(center, width, height, rotation):
        """Draw a rectangle, centered at x, y.

        Arguments:
        x (int/float):
            The x coordinate of the center of the shape.
        y (int/float):
            The y coordinate of the center of the shape.
        width (int/float):
            The width of the rectangle.
        height (int/float):
            The height of the rectangle.
        """
        x, y = center

        points = []

        # The distance from the center of the rectangle to
        # one of the corners is the same for each corner.
        radius = math.sqrt((height / 2) ** 2 + (width / 2) ** 2)

        # Get the angle to one of the corners with respect
        # to the x-axis.
        angle = math.atan2(height / 2, width / 2)

        # Adjust angles for Pygame, where 0 angle is to the right
        # and rotations are clockwise
        angles = [
            angle - math.pi / 2 + rotation,
            math.pi - angle - math.pi / 2 + rotation,
            math.pi + angle - math.pi / 2 + rotation,
            -angle - math.pi / 2 + rotation,
        ]

        # Calculate the coordinates of each corner for Pygame
        for angle in angles:
            x_offset = radius * math.cos(angle)
            y_offset = radius * math.sin(angle)  # Invert y-coordinate
            points.append((x + x_offset, y + y_offset))

        return points

    @staticmethod
    def get_endpoints(center, map_obj):
        center_pos = center
        length = map_obj[2]  # Already half the length
        yaw = map_obj[5]

        start = center_pos - np.array([length * np.cos(yaw), length * np.sin(yaw)])
        end = center_pos + np.array([length * np.cos(yaw), length * np.sin(yaw)])
        return start, end
    
    def init_map(self):
        """Initialize the static map elements."""
        self.map_surf = self.surf.copy()  # Create a copy of the main surface to hold the map
        self.map_surf.fill(self.BACKGROUND_COLOR)

        map_info = (
            self.sim.map_observation_tensor()
            .to_torch()[self.world_render_idx]
            .cpu()
            .numpy()
        )

        map_info = map_info[map_info[:, -1] != float(gpudrive.EntityType.Padding)]

        for idx, map_obj in enumerate(map_info):
            if map_obj[-1] == float(gpudrive.EntityType.Padding):
                continue
            elif map_obj[-1] <= float(gpudrive.EntityType.RoadLane):
                start, end = PyGameVisualizer.get_endpoints(map_obj[:2], map_obj)
                start = self.scale_coords(start)
                end = self.scale_coords(end)
                pygame.draw.line(self.map_surf, self.color_dict[map_obj[-1]], start, end, 2)
            elif map_obj[-1] <= float(gpudrive.EntityType.StopSign):
                center, width, height, rotation = (
                    map_obj[:2],
                    map_obj[3],
                    map_obj[2],
                    map_obj[5],
                )
                if map_obj[-1] == float(gpudrive.EntityType.StopSign):
                    width *= self.zoom_scale_x
                    height *= self.zoom_scale_y
                box_corners = PyGameVisualizer.compute_agent_corners(
                    center, width, height, rotation
                )
                for i, box_corner in enumerate(box_corners):
                    box_corners[i] = self.scale_coords(box_corner)
                pygame.draw.polygon(
                    surface=self.map_surf,
                    color=self.color_dict[map_obj[-1]],
                    points=box_corners,
                )

    def draw(self, cont_agent_mask):
        """Render the environment."""
        self.surf.fill(self.BACKGROUND_COLOR)
        self.surf.blit(self.map_surf, (0, 0))
        # Get agent info
        agent_info = (
            self.sim.absolute_self_observation_tensor()
            .to_torch()[self.world_render_idx, :, :]
            .cpu()
            .detach()
            .numpy()
        )

        # Get the agent goal positions and current positions
        agent_pos = agent_info[:, :2]  # x, y
        goal_pos = agent_info[:, 8:10]  # x, y
        agent_rot = agent_info[:, 7]  # heading
        agent_sizes = agent_info[:, 10:12]  # length, width

        num_agents_in_scene = np.count_nonzero(goal_pos[:, 0])

        # Draw the agent positions
        for agent_idx in range(num_agents_in_scene):
            info_tensor = self.sim.info_tensor().to_torch()[self.world_render_idx]
            if info_tensor[agent_idx, -1] == float(gpudrive.EntityType.Padding) or info_tensor[agent_idx, -1] == float(gpudrive.EntityType._None):
                continue

            agent_corners = PyGameVisualizer.compute_agent_corners(
                agent_pos[agent_idx],
                agent_sizes[agent_idx, 1],
                agent_sizes[agent_idx, 0],
                agent_rot[agent_idx],
            )

            for i, agent_corner in enumerate(agent_corners):
                agent_corners[i] = self.scale_coords(agent_corner)

            current_goal_scaled = self.scale_coords(goal_pos[agent_idx])

            mod_idx = agent_idx % len(self.COLOR_LIST)

            if cont_agent_mask[self.world_render_idx, agent_idx]:
                mod_idx = 0

            pygame.draw.polygon(
                surface=self.surf,
                color=self.COLOR_LIST[mod_idx],
                points=agent_corners,
            )

            pygame.draw.circle(
                surface=self.surf,
                color=self.COLOR_LIST[mod_idx],
                center=(
                    int(current_goal_scaled[0]),
                    int(current_goal_scaled[1]),
                ),
                radius=self.goal_radius,
            )

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return PyGameVisualizer._create_image_array(self.surf)
        else:
            return self.isopen

    @staticmethod
    def _create_image_array(surf):
        return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))

    def destroy(self):
        pygame.display.quit()
        pygame.quit()
