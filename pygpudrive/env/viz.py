import pygame
from pygame import Color
import numpy as np
from pygame.sprite import Sprite
import os
import math
import gpudrive

from pygpudrive.env.config import MadronaOption, PygameOption, RenderMode


class PyGameVisualizer:
    WINDOW_W, WINDOW_H = 1920, 1080
    BACKGROUND_COLOR = (0, 0, 0)
    PADDING_PCT = 0.1
    COLOR_LIST = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 165, 0),  # Orange
    ]
    color_dict = {
        float(gpudrive.EntityType.RoadEdge): (255, 255, 255),  # Black
        float(gpudrive.EntityType.RoadLane): (225, 225, 225),  # Grey
        float(gpudrive.EntityType.RoadLine): (225, 255, 225),  # Green
        float(gpudrive.EntityType.SpeedBump): (255, 0, 255),  # Red
        float(gpudrive.EntityType.CrossWalk): (213, 20, 20),  # dark
        float(gpudrive.EntityType.StopSign): (255, 0, 255),  # Blue
    }

    def __init__(self, sim, world_render_idx, render_config, goal_radius):
        self.sim = sim
        self.world_render_idx = world_render_idx
        self.render_config = render_config
        self.goal_radius = goal_radius

        self.padding_x = self.PADDING_PCT * self.WINDOW_W
        self.padding_y = self.PADDING_PCT * self.WINDOW_H

        if self.render_config.render_mode in {RenderMode.PYGAME_ABSOLUTE, RenderMode.PYGAME_EGOCENTRIC, RenderMode.PYGAME_LIDAR}:
            pygame.init()
            pygame.font.init()
            self.screen = None
            self.clock = None
            if self.screen is None and self.render_config.view_option == PygameOption.HUMAN:
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.WINDOW_W, self.WINDOW_H))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.surf = pygame.Surface((self.WINDOW_W, self.WINDOW_H))
            self.compute_window_settings()
            # self.init_map()

    def compute_window_settings(self, map_info = None):
        if map_info is None:
            map_info = (
                self.sim.map_observation_tensor()
                .to_torch()[self.world_render_idx]
                .cpu()
                .numpy()
            )

        # Adjust window dimensions by subtracting padding
        adjusted_window_width = self.WINDOW_W - self.padding_x
        adjusted_window_height = self.WINDOW_H - self.padding_y

        self.zoom_scale_x = adjusted_window_width / (
            map_info[:, 0].max() - map_info[:, 0].min()
        )
        self.zoom_scale_y = adjusted_window_height / (
            map_info[:, 1].max() - map_info[:, 1].min()
        )

        self.window_center = np.mean(map_info, axis=0)
        

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
            + (self.WINDOW_W / 2)
            + self.padding_x / 2
        )
        y_scaled = (
            (y - self.window_center[1]) * self.zoom_scale_y
            + (self.WINDOW_H / 2)
            + self.padding_y / 2
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
    
    def draw_map(self, surf, map_info):
        for idx, map_obj in enumerate(map_info):
            if map_obj[-1] == float(gpudrive.EntityType._None):
                continue
            elif map_obj[-1] <= float(gpudrive.EntityType.RoadLane):
                start, end = PyGameVisualizer.get_endpoints(map_obj[:2], map_obj)
                start = self.scale_coords(start)
                end = self.scale_coords(end)
                pygame.draw.line(surf, self.color_dict[map_obj[-1]], start, end, 2)
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
                    surface=surf,
                    color=self.color_dict[map_obj[-1]],
                    points=box_corners,
                )

    def init_map(self):
        """Initialize the static map elements."""

        if(self.render_config.render_mode == RenderMode.PYGAME_EGOCENTRIC):
            return
        self.map_surf = self.surf.copy()  # Create a copy of the main surface to hold the map

        map_info = (
            self.sim.map_observation_tensor()
            .to_torch()[self.world_render_idx]
            .cpu()
            .numpy()
        )

        self.draw_map(self.map_surf, map_info)

        

    def getRender(self, **kwargs):
        if self.render_config.render_mode in {RenderMode.PYGAME_ABSOLUTE, RenderMode.PYGAME_EGOCENTRIC, RenderMode.PYGAME_LIDAR}:
            cont_agent_mask = kwargs.get('cont_agent_mask', None)
            return self.draw(cont_agent_mask)
        elif self.render_config.render_mode == RenderMode.MADRONA_RGB:
            if(self.render_config.view_option == MadronaOption.TOP_DOWN):
                raise NotImplementedError
            return self.sim.rgb_tensor().to_torch()
        elif self.render_config.render_mode == RenderMode.MADRONA_DEPTH:
            if(self.render_config.view_option == MadronaOption.TOP_DOWN):
                raise NotImplementedError
            return self.sim.depth_tensor().to_torch()

    def draw(self, cont_agent_mask):
        """Render the environment."""

        if self.render_config.render_mode == RenderMode.PYGAME_EGOCENTRIC:
            render_rgbs = []
            render_mask = self.create_render_mask()
            num_agents = render_mask.sum().item()
            # Loop through each agent to render their egocentric view
            for agent_idx in range(num_agents):
                self.surf.fill(self.BACKGROUND_COLOR)
                if(not render_mask[agent_idx]):
                    continue
                agent_map_info = (
                    self.sim.agent_roadmap_tensor()
                    .to_torch()[self.world_render_idx, agent_idx, :, :]
                    .cpu()
                    .detach()
                    .numpy()
                )
                agent_map_info = agent_map_info[(agent_map_info[:, -1] != 0.0) & (agent_map_info[:, -1] != 10.0)]

                agent_info = (
                    self.sim.self_observation_tensor()
                    .to_torch()[self.world_render_idx, agent_idx, :]
                    .cpu()
                    .detach()
                    .numpy()
                )

                partner_agent_info = (
                    self.sim.partner_observations_tensor()
                    .to_torch()[self.world_render_idx, agent_idx, :, :]
                    .cpu()
                    .detach()
                    .numpy()
                )
                partner_agent_info = partner_agent_info[partner_agent_info[:, -1] == 7.0]
                
                goal_pos = agent_info[3:5]  # x, y
                agent_size = agent_info[1:3]  # length, width

                # Create a temporary surface for the egocentric view
                temp_surf = pygame.Surface((self.surf.get_width(), self.surf.get_height()))
                temp_surf.fill(self.BACKGROUND_COLOR)

                self.draw_map(temp_surf, agent_map_info)
                # Transform the map surface to the agent's egocentric view
                agent_corners = PyGameVisualizer.compute_agent_corners(
                    (0,0),
                    agent_size[1],
                    agent_size[0],
                    0
                )
                agent_corners = [self.scale_coords(corner) for corner in agent_corners]
                current_goal_scaled = self.scale_coords(goal_pos)

                pygame.draw.polygon(
                    surface=temp_surf,
                    color=self.COLOR_LIST[0],
                    points=agent_corners,
                )

                pygame.draw.circle(
                    surface=temp_surf,
                    color=self.COLOR_LIST[0],
                    center=(
                        int(current_goal_scaled[0]),
                        int(current_goal_scaled[1]),
                    ),
                    radius= self.goal_radius * self.zoom_scale_x,
                )

                for agent in partner_agent_info:
                    agent_pos = agent[1:3]
                    agent_rot = agent[3]
                    agent_size = agent[4:6]

                    agent_corners = PyGameVisualizer.compute_agent_corners(
                        agent_pos,
                        agent_size[1],
                        agent_size[0],
                        agent_rot,
                    )

                    agent_corners = [self.scale_coords(corner) for corner in agent_corners]

                    pygame.draw.polygon(
                        surface=temp_surf,
                        color=self.COLOR_LIST[1],
                        points=agent_corners,
                    )

                #blit temp surf on self.surf
                self.surf.blit(temp_surf, (0, 0))
                # Capture the RGB array for the agent's view
                render_rgbs.append(PyGameVisualizer._create_image_array(self.surf))

            return render_rgbs
        elif self.render_config.render_mode == RenderMode.PYGAME_ABSOLUTE:
            render_mask = self.create_render_mask()
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
                if not render_mask[agent_idx]:
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
                    radius=self.goal_radius * self.zoom_scale_x,
                )

            if self.render_config.view_option == PygameOption.HUMAN:
                pygame.event.pump()
                self.clock.tick(self.metadata["render_fps"])
                assert self.screen is not None
                self.screen.fill(0)
                self.screen.blit(self.surf, (0, 0))
                pygame.display.flip()
            elif self.render_config.view_option == PygameOption.RGB:
                return PyGameVisualizer._create_image_array(self.surf)
            else:
                return self.isopen
        elif self.render_config.render_mode == RenderMode.PYGAME_LIDAR:
            render_rgbs = []
            render_mask = self.create_render_mask()
            num_agents = render_mask.sum().item()
            # Loop through each agent to render their egocentric view
            for agent_idx in range(num_agents):
                self.surf.fill(self.BACKGROUND_COLOR)
                temp_surf = pygame.Surface((self.surf.get_width(), self.surf.get_height()))
                temp_surf.fill(self.BACKGROUND_COLOR)

                agent_info = (
                    self.sim.self_observation_tensor()
                    .to_torch()[self.world_render_idx, agent_idx, :]
                    .cpu()
                    .detach()
                    .numpy()
                )

                numLidarSamples = 1024

                lidar_data = (
                    self.sim.lidar_tensor()
                    .to_torch()[self.world_render_idx, agent_idx, :, :]
                    .cpu()
                    .detach()
                    .numpy()
                )

                lidar_depths = lidar_data[:, 0]

                lidar_angles = np.linspace(0, 2 * np.pi, numLidarSamples)

                num_lidar_plotted = 0

                for i in range(numLidarSamples):
                    angle = lidar_angles[i]
                    depth = lidar_depths[i]
                    if(depth == 0):
                        continue
                    x = depth * np.cos(angle)
                    y = depth * np.sin(angle)

                    start = self.scale_coords((0,0))
                    end = self.scale_coords(np.array([x, y]))

                    pygame.draw.circle(
                        surface=temp_surf,
                        color=(255, 255, 255),
                        center=(
                            int(end[0]),
                            int(end[1]),
                        ),
                        radius= 2,
                    )
                    # pygame.draw.line(temp_surf, (255, 255, 255), start, end, 2)
                    num_lidar_plotted += 1
                
                goal_pos = agent_info[3:5]  # x, y
                agent_size = agent_info[1:3]  # length, width

                agent_corners = PyGameVisualizer.compute_agent_corners(
                    (0,0),
                    agent_size[1],
                    agent_size[0],
                    np.pi/2
                )
                agent_corners = [self.scale_coords(corner) for corner in agent_corners]
                current_goal_scaled = self.scale_coords(goal_pos)

                pygame.draw.polygon(
                    surface=temp_surf,
                    color=self.COLOR_LIST[0],
                    points=agent_corners,
                )

                pygame.draw.circle(
                    surface=temp_surf,
                    color=self.COLOR_LIST[0],
                    center=(
                        int(current_goal_scaled[0]),
                        int(current_goal_scaled[1]),
                    ),
                    radius= self.goal_radius * self.zoom_scale_x,
                )

                #blit temp surf on self.surf
                self.surf.blit(temp_surf, (0, 0))
                # Capture the RGB array for the agent's view
                render_rgbs.append(PyGameVisualizer._create_image_array(self.surf))
            return render_rgbs
                
    @staticmethod
    def _create_image_array(surf):
        return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))

    def destroy(self):
        pygame.display.quit()
        pygame.quit()
