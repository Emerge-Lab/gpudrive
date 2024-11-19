import pygame
import pygame.gfxdraw
import numpy as np
import math
import gpudrive

from pygpudrive.env.config import MadronaOption, PygameOption, RenderMode

# AGENT COLORS
PINK = (255, 105, 180)
GREEN = (113, 228, 0)
BLUE = (0, 127, 255)
DODGER_BLUE = (30, 144, 255)
RED_ORANGE = (255, 69, 0)
WHITE = (255, 255, 255)
CHARCOAL = (22, 28, 32) 
BLACK = (3, 3, 3)
GRAY_DARKER = (84, 82, 82)
YELLOW = (232, 232, 9)

STATIC_AGENT_ID = 2


class PyGameVisualizer:
    WINDOW_W, WINDOW_H = 1920, 1080
    PADDING_PCT = 0.0
    COLOR_LIST = [
        BLUE,
        PINK,
        GREEN,
        DODGER_BLUE,
        RED_ORANGE,
    ]

    def __init__(self, sim, render_config, goal_radius):
        self.sim = sim
        self.render_config = render_config
        self.WINDOW_W, self.WINDOW_H = render_config.resolution
        self.goal_radius = goal_radius
        
        if self.render_config.color_scheme == "light":
            self.BACKGROUND_COLOR = WHITE
            self.vehicle_idx_color = CHARCOAL
        else:
            self.BACKGROUND_COLOR = CHARCOAL
            self.vehicle_idx_color = WHITE
        
        # ROAD MAP COLORS
        self.color_dict = {
            float(gpudrive.EntityType.RoadEdge): BLACK if self.render_config.color_scheme == "dark" else (47,79,79),
            float(gpudrive.EntityType.RoadLine): GRAY_DARKER,  # Yellow
            float(gpudrive.EntityType.RoadLane): (225, 225, 225),  # Grey
            float(gpudrive.EntityType.SpeedBump): (138, 43, 226),  # Purple
            float(gpudrive.EntityType.CrossWalk): (255, 255, 255),  # White
            float(gpudrive.EntityType.StopSign): (213, 20, 20),  # Dark red
            float(gpudrive.EntityType.Vehicle): (0, 255, 0),  # Green
            float(gpudrive.EntityType.Pedestrian): (0, 255, 0),  # Green
            float(gpudrive.EntityType.Cyclist): (0, 0, 255),  # Blue
        }

        self.num_agents = self.sim.shape_tensor().to_torch().cpu().numpy()
        self.num_worlds = self.sim.shape_tensor().to_torch().shape[0]

        self.padding_x = self.PADDING_PCT * self.WINDOW_W
        self.padding_y = self.PADDING_PCT * self.WINDOW_H

        self.zoom_scales_x = np.array([1.0] * self.num_worlds)
        self.zoom_scales_y = np.array([1.0] * self.num_worlds)
        self.window_centers = np.array([[0, 0]] * self.num_worlds)

        if self.render_config.render_mode in {
            RenderMode.PYGAME_ABSOLUTE,
            RenderMode.PYGAME_EGOCENTRIC,
            RenderMode.PYGAME_LIDAR,
        }:
            pygame.init()
            pygame.font.init()
            self.screen = None
            self.clock = None
            if (
                self.screen is None
                and self.render_config.view_option == PygameOption.HUMAN
            ):
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.WINDOW_W, self.WINDOW_H)
                )
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.surf = pygame.Surface((self.WINDOW_W, self.WINDOW_H))
            self.compute_window_settings()

    @staticmethod
    def get_all_endpoints(map_info):
        centers = map_info[:, :2]
        lengths = map_info[:, 2]
        yaws = map_info[:, 5]

        offsets = np.column_stack(
            (lengths * np.cos(yaws), lengths * np.sin(yaws))
        )
        starts = centers - offsets
        ends = centers + offsets
        return starts, ends

    def draw_line(self, surf, start, end, color, thickness=1, fill_shape=True):
        c1 = (start[0] + end[0]) / 2
        c2 = (start[1] + end[1]) / 2
        center_L1 = (c1, c2)
        length = math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
        angle = math.atan2(start[1] - end[1], start[0] - end[0])

        UL = (
            center_L1[0]
            + (length / 2.0) * np.cos(angle)
            - (thickness / 2.0) * np.sin(angle),
            center_L1[1]
            + (thickness / 2.0) * np.cos(angle)
            + (length / 2.0) * np.sin(angle),
        )
        UR = (
            center_L1[0]
            - (length / 2.0) * np.cos(angle)
            - (thickness / 2.0) * np.sin(angle),
            center_L1[1]
            + (thickness / 2.0) * np.cos(angle)
            - (length / 2.0) * np.sin(angle),
        )
        BL = (
            center_L1[0]
            + (length / 2.0) * np.cos(angle)
            + (thickness / 2.0) * np.sin(angle),
            center_L1[1]
            - (thickness / 2.0) * np.cos(angle)
            + (length / 2.0) * np.sin(angle),
        )
        BR = (
            center_L1[0]
            - (length / 2.0) * np.cos(angle)
            + (thickness / 2.0) * np.sin(angle),
            center_L1[1]
            - (thickness / 2.0) * np.cos(angle)
            - (length / 2.0) * np.sin(angle),
        )

        if fill_shape:
            pygame.gfxdraw.aapolygon(surf, (UL, UR, BR, BL), color)
            pygame.gfxdraw.filled_polygon(surf, (UL, UR, BR, BL), color)
        else:
            pygame.gfxdraw.aapolygon(surf, (UL, UR, BR, BL), color)

    def draw_circle(self, surf, center, radius, color, thickness=3):
        for i in range(thickness):
            try:
                pygame.gfxdraw.aacircle(
                    surf,
                    int(center[0]),
                    int(center[1]),
                    int(radius) + i,
                    color,
                )
            except:
                continue

    def compute_window_settings(self, map_infos=None):
        if map_infos is None:
            map_infos = (
                self.sim.map_observation_tensor().to_torch().cpu().numpy()
            )
        assert map_infos.shape[0] <= self.num_worlds
        for i in range(map_infos.shape[0]):
            map_info = map_infos[i]
            map_info = map_info[
                map_info[:, 6] != float(gpudrive.EntityType.Padding)
            ]
            roads = map_info[
                map_info[:, 6] <= float(gpudrive.EntityType.RoadLane)
            ]
            endpoints = PyGameVisualizer.get_all_endpoints(roads)

            all_endpoints = np.concatenate(endpoints, axis=0)

            # Adjust window dimensions by subtracting padding
            adjusted_window_width = self.WINDOW_W - self.padding_x
            adjusted_window_height = self.WINDOW_H - self.padding_y

            self.zoom_scales_x[i] = adjusted_window_width / (
                all_endpoints[:, 0].max() - all_endpoints[:, 0].min()
            )
            self.zoom_scales_y[i] = adjusted_window_height / (
                all_endpoints[:, 1].max() - all_endpoints[:, 1].min()
            )

            self.window_centers[i] = np.array(
                [
                    (all_endpoints[:, 0].max() + all_endpoints[:, 0].min())
                    / 2,
                    (all_endpoints[:, 1].max() + all_endpoints[:, 1].min())
                    / 2,
                ]
            )

    def scale_coords(self, coords, world_render_idx):
        """Scale the coordinates to fit within the pygame surface window and center them.
        Args:
            coords: x, y coordinates
        """
        x, y = coords
        x_scaled = (
            (x - self.window_centers[world_render_idx][0])
            * self.zoom_scales_x[world_render_idx]
            + self.WINDOW_W / 2
            - self.padding_x / 2
        )
        y_scaled = (
            (y - self.window_centers[world_render_idx][1])
            * self.zoom_scales_y[world_render_idx]
            + self.WINDOW_H / 2
            - self.padding_y / 2
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

        start = center_pos - np.array(
            [length * np.cos(yaw), length * np.sin(yaw)]
        )
        end = center_pos + np.array(
            [length * np.cos(yaw), length * np.sin(yaw)]
        )
        return start, end

    def draw_map(self, surf, map_info, world_render_idx=0):
        """Draw static map elements."""
        for idx, map_obj in enumerate(map_info):

            if map_obj[6] == float(gpudrive.EntityType.Padding) or map_obj[6] == float(gpudrive.EntityType._None):
                continue

            elif map_obj[6] <= float(gpudrive.EntityType.RoadLane):
                start, end = PyGameVisualizer.get_endpoints(
                    map_obj[:2], map_obj
                )
                start = self.scale_coords(start, world_render_idx)
                end = self.scale_coords(end, world_render_idx)

                # DRAW ROAD EDGE
                if map_obj[6] == float(gpudrive.EntityType.RoadEdge):
                    self.draw_line(
                        surf,
                        start,
                        end,
                        self.color_dict[map_obj[6]],
                        thickness=self.render_config.line_thickness,
                    )

                # DRAW ROAD LINES/LANES
                else:
                    self.draw_line(
                        surf,
                        start,
                        end,
                        self.color_dict[map_obj[6]],
                        thickness=self.render_config.line_thickness,
                    )

            # DRAW STOP SIGNS
            elif map_obj[6] <= float(gpudrive.EntityType.StopSign):

                center, width, height, rotation = (
                    map_obj[:2],
                    map_obj[3],
                    map_obj[2],
                    map_obj[5],
                )
                if map_obj[6] == float(gpudrive.EntityType.StopSign):

                    width *= self.zoom_scales_x[world_render_idx]
                    height *= self.zoom_scales_y[world_render_idx]

                box_corners = PyGameVisualizer.compute_agent_corners(
                    center, width, height, rotation
                )
                for i, box_corner in enumerate(box_corners):
                    box_corners[i] = self.scale_coords(
                        box_corner, world_render_idx
                    )
                # if map_obj[6] == float(gpudrive.EntityType.SpeedBump):
                #     pygame.gfxdraw.aapolygon(
                #         surf, box_corners, self.color_dict[map_obj[6]]
                #     )
                #     #to draw stripes
                #     scaled_width = box_corners[1]-box_corners[0]
                #     scaled_heigth = box_corners[2] - box_corners[3]
                #     scaled_dashed_length = scaled_width/4
                #     for i in range(0, 2):
                #         box_corners[0] = tuple(np.add(box_corner[0]) + (scaled_dashed_length,0) )  
                #         box_corners[1] = box_corners[0] + scaled_dashed_length
                #         box_corners[2]                       
                #         pygame.gfxdraw.filled_polygon(
                #            surf, box_corners, self.color_dict[map_obj[6]]
                #         )
                if map_obj[6] == float(gpudrive.EntityType.SpeedBump):
                    # Draw the outer rectangle (the speed bump)
                    pygame.gfxdraw.aapolygon(
                        surf, box_corners, BLACK #self.color_dict[map_obj[6]]
                    )
                    # pygame.gfxdraw.filled_polygon(
                    #     surf, box_corners, BLACK
                    # )

                     # Compute the top edge length (between box_corners[0] and box_corners[1])
                    top_edge_length = math.sqrt(
                        (box_corners[1][0] - box_corners[0][0]) ** 2 +
                        (box_corners[1][1] - box_corners[0][1]) ** 2
                    )

                    # Compute the height using the vertical difference (between box_corners[0] and box_corners[3])
                    effective_height = math.sqrt(
                        (box_corners[3][0] - box_corners[0][0]) ** 2 +
                        (box_corners[3][1] - box_corners[0][1]) ** 2
                    )

                    # Stripe properties
                    num_stripes = 4  # Number of stripes
                    stripe_width = top_edge_length / num_stripes  

                    # Vector along the top edge (unit vector)
                    top_edge_vector = (
                        (box_corners[1][0] - box_corners[0][0]) / top_edge_length,
                        (box_corners[1][1] - box_corners[0][1]) / top_edge_length,
                    )

                    height_vector = (
                        (box_corners[3][0] - box_corners[0][0]) / effective_height,
                        (box_corners[3][1] - box_corners[0][1]) / effective_height,
                    )

                    # Draw each stripe
                    for i in range(num_stripes):
                        # Start and end points of each stripe along the top edge
                        stripe_start = (
                            box_corners[0][0] + i * stripe_width * top_edge_vector[0],
                            box_corners[0][1] + i * stripe_width * top_edge_vector[1],
                        )
                        stripe_end = (
                            stripe_start[0] + stripe_width * top_edge_vector[0],
                            stripe_start[1] + stripe_width * top_edge_vector[1],
                        )

                        stripe_corners = [
                            stripe_start,
                            stripe_end,
                            (
                                stripe_end[0] + effective_height * height_vector[0],
                                stripe_end[1] + effective_height * height_vector[1],
                            ),
                            (
                                stripe_start[0] + effective_height * height_vector[0],
                                stripe_start[1] + effective_height * height_vector[1],
                            ),
                        ]

                        # Draw the stripe
                        pygame.gfxdraw.aapolygon(surf, stripe_corners, BLACK)
                        stripe_color = BLACK if i % 2 == 0 else YELLOW
                        pygame.gfxdraw.filled_polygon(surf, stripe_corners, stripe_color)

                  
                        # pygame.gfxdraw.filled_polygon(
                        #     surf, box_corners, PINK
                        # )


                    
                else:
                    pygame.gfxdraw.aapolygon(
                        surf, box_corners, self.color_dict[map_obj[6]]
                    )
                    pygame.gfxdraw.filled_polygon(
                        surf, box_corners, self.color_dict[map_obj[6]]
                    )

    def init_map(self):
        """Initialize the static map elements."""

        if self.render_config.render_mode == RenderMode.PYGAME_EGOCENTRIC:
            return

        self.map_surf.fill(self.BACKGROUND_COLOR)
        self.map_surfs = []
        for i in range(self.num_worlds):
            map_surf = self.surf.copy()
            map_info = (
                self.sim.map_observation_tensor().to_torch()[i].cpu().numpy()
            )
            self.draw_map(map_surf, map_info, i)
            self.map_surfs.append(map_surf)

    def getRender(
        self, world_render_idx=0, color_objects_by_actor=None, **kwargs
    ):
        if self.render_config.render_mode in {
            RenderMode.PYGAME_ABSOLUTE,
            RenderMode.PYGAME_EGOCENTRIC,
            RenderMode.PYGAME_LIDAR,
        }:
            cont_agent_mask = kwargs.get("cont_agent_mask", None)
            return self.draw(
                cont_agent_mask, world_render_idx, color_objects_by_actor
            )
        elif self.render_config.render_mode == RenderMode.MADRONA_RGB:
            if self.render_config.view_option == MadronaOption.TOP_DOWN:
                raise NotImplementedError
            return self.sim.rgb_tensor().to_torch()
        elif self.render_config.render_mode == RenderMode.MADRONA_DEPTH:
            if self.render_config.view_option == MadronaOption.TOP_DOWN:
                raise NotImplementedError
            return self.sim.depth_tensor().to_torch()

    def plotLidar(self, surf, lidar_data, world_render_idx):
        numLidarSamples = lidar_data.shape[0]

        lidar_entity_types = lidar_data[:, 1]
        lidar_pos = lidar_data[:, 2:4]

        for i in range(numLidarSamples):
            if(lidar_entity_types[i] == float(gpudrive.EntityType._None)):
                continue
            coords = lidar_pos[i]
            scaled_coords = self.scale_coords(coords, world_render_idx)
           
            pygame.gfxdraw.aacircle(
                surf,
                int(scaled_coords[0]),
                int(scaled_coords[1]),
                2,
                self.color_dict[lidar_entity_types[i]]
            )

            pygame.gfxdraw.filled_circle(
                surf,
                int(scaled_coords[0]),
                int(scaled_coords[1]),
                2,
                self.color_dict[lidar_entity_types[i]]
            )

    def draw(
        self, cont_agent_mask, world_render_idx=0, color_objects_by_actor=None
    ):
        """Render the environment."""

        if self.render_config.render_mode == RenderMode.PYGAME_EGOCENTRIC:
            render_rgbs = []
            num_agents = self.num_agents[world_render_idx][0]
            # Loop through each agent to render their egocentric view
            for agent_idx in range(num_agents):
                info_tensor = self.sim.info_tensor().to_torch()[
                    world_render_idx
                ]
                if info_tensor[agent_idx, 6] == float(
                    gpudrive.EntityType.Padding
                ) or info_tensor[agent_idx, 6] == float(
                    gpudrive.EntityType._None
                ):
                    continue

                self.surf.fill(self.BACKGROUND_COLOR)
                agent_map_info = (
                    self.sim.agent_roadmap_tensor()
                    .to_torch()[world_render_idx, agent_idx, :, :]
                    .cpu()
                    .detach()
                    .numpy()
                )
                agent_map_info = agent_map_info[
                    (agent_map_info[:, 6] != 0.0)
                    & (agent_map_info[:, 6] != 10.0)
                ]

                agent_info = (
                    self.sim.self_observation_tensor()
                    .to_torch()[world_render_idx, agent_idx, :]
                    .cpu()
                    .detach()
                    .numpy()
                )

                partner_agent_info = (
                    self.sim.partner_observations_tensor()
                    .to_torch()[world_render_idx, agent_idx, :, :]
                    .cpu()
                    .detach()
                    .numpy()
                )
                partner_agent_info = partner_agent_info[
                    partner_agent_info[:, -1] == 7.0
                ]

                goal_pos = agent_info[3:5]  # x, y
                agent_size = agent_info[1:3]  # length, width

                # Create a temporary surface for the egocentric view
                temp_surf = pygame.Surface(
                    (self.surf.get_width(), self.surf.get_height())
                )
                temp_surf.fill(self.BACKGROUND_COLOR)

                self.draw_map(temp_surf, agent_map_info)
                # Transform the map surface to the agent's egocentric view
                agent_corners = PyGameVisualizer.compute_agent_corners(
                    (0, 0), agent_size[1], agent_size[0], 0
                )
                agent_corners = [
                    self.scale_coords(corner, world_render_idx)
                    for corner in agent_corners
                ]
                current_goal_scaled = self.scale_coords(
                    goal_pos, world_render_idx
                )

                pygame.gfxdraw.aapolygon(
                    temp_surf, agent_corners, self.COLOR_LIST[0]
                )
                pygame.gfxdraw.filled_polygon(
                    temp_surf, agent_corners, self.COLOR_LIST[0]
                )

                self.draw_circle(
                    temp_surf,
                    current_goal_scaled,
                    self.goal_radius * self.zoom_scales_x[world_render_idx],
                    self.COLOR_LIST[0],
                )

                for agent in partner_agent_info:
                    agent_pos = agent[1:3]
                    agent_rot = agent[3]
                    agent_size = agent[4:6]
                    agent_type = agent[6]

                    agent_corners = PyGameVisualizer.compute_agent_corners(
                        agent_pos,
                        agent_size[1],
                        agent_size[0],
                        agent_rot,
                    )

                    agent_corners = [
                        self.scale_coords(corner, world_render_idx)
                        for corner in agent_corners
                    ]

                    pygame.gfxdraw.aapolygon(
                        temp_surf, agent_corners, self.COLOR_LIST[1]
                    )
                    pygame.gfxdraw.filled_polygon(
                        temp_surf,
                        agent_corners,
                        self.COLOR_LIST[1],
                    )

                # blit temp surf on self.surf
                self.surf.blit(temp_surf, (0, 0))
                # Capture the RGB array for the agent's view
                render_rgbs.append(
                    PyGameVisualizer._create_image_array(self.surf)
                )

            return render_rgbs
        elif self.render_config.render_mode == RenderMode.PYGAME_ABSOLUTE:
            self.surf.fill(self.BACKGROUND_COLOR)
            map_info = (
                self.sim.map_observation_tensor()
                .to_torch()[world_render_idx]
                .cpu()
                .numpy()
            )
            self.draw_map(self.surf, map_info, world_render_idx)
            # Get agent info
            agent_info = (
                self.sim.absolute_self_observation_tensor()
                .to_torch()[world_render_idx, :, :]
                .cpu()
                .detach()
                .numpy()
            )

            # Get the agent goal positions and current positions
            agent_pos = agent_info[:, :2]  # x, y
            goal_pos = agent_info[:, 8:10]  # x, y
            agent_rot = agent_info[:, 7]  # heading
            agent_sizes = agent_info[:, 10:12]  # length, width
            agent_response_types = (  # 0: Valid (can be controlled), 2: Invalid (static vehicles)
                self.sim.response_type_tensor()
                .to_torch()[world_render_idx, :, :]
                .cpu()
                .detach()
                .numpy()
            )

            num_agents = self.num_agents[world_render_idx][0]
            if color_objects_by_actor is not None:
                categories = list(color_objects_by_actor.keys())

            valid_agent_indices = list(
                range((agent_response_types == 0).sum())
            )

            # Draw the agent positions
            for agent_idx in range(num_agents):

                if color_objects_by_actor is not None:
                    for i in range(len(categories)):
                        if agent_idx in color_objects_by_actor[categories[i]]:
                            mod_idx = i
                            break
                    else:
                        mod_idx = 3
                else:
                    mod_idx = agent_idx % len(self.COLOR_LIST)

                color = self.COLOR_LIST[mod_idx]

                info_tensor = self.sim.info_tensor().to_torch()[
                    world_render_idx
                ]
                if info_tensor[agent_idx, -1] == float(
                    gpudrive.EntityType.Padding
                ) or info_tensor[agent_idx, -1] == float(
                    gpudrive.EntityType._None
                ):
                    continue

                agent_corners = PyGameVisualizer.compute_agent_corners(
                    agent_pos[agent_idx],
                    agent_sizes[agent_idx, 1],
                    agent_sizes[agent_idx, 0],
                    agent_rot[agent_idx],
                )

                for i, agent_corner in enumerate(agent_corners):
                    agent_corners[i] = self.scale_coords(
                        agent_corner, world_render_idx
                    )

                current_goal_scaled = self.scale_coords(
                    goal_pos[agent_idx], world_render_idx
                )

                # Agent is static
                if agent_response_types[agent_idx] == STATIC_AGENT_ID:
                    color = (128, 128, 128)

                pygame.gfxdraw.aapolygon(self.surf, agent_corners, color)
                pygame.gfxdraw.filled_polygon(self.surf, agent_corners, color)

                # Draw object indices for the controllable agents
                if (
                    self.render_config.draw_obj_idx
                    and agent_response_types[agent_idx] != STATIC_AGENT_ID
                ):
                    scaled_font_size = (
                        self.render_config.obj_idx_font_size
                        * int(self.zoom_scales_x[world_render_idx])
                    )
                    font = pygame.font.Font(None, scaled_font_size)
                    text = font.render(
                        str(valid_agent_indices.pop(0)), True, self.vehicle_idx_color
                    )
                    text_rect = text.get_rect(
                        center=(
                            agent_corners[0][0]
                            - 2 * int(self.zoom_scales_y[0]),
                            agent_corners[0][1],
                        )
                    )
                    self.surf.blit(text, text_rect)

                if agent_response_types[agent_idx] != STATIC_AGENT_ID:
                    self.draw_circle(
                        self.surf,
                        current_goal_scaled,
                        self.goal_radius
                        * self.zoom_scales_x[world_render_idx],
                        color,
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
            num_agents = self.num_agents[world_render_idx][0]

            # Loop through each agent to render their egocentric view
            for agent_idx in range(num_agents):
                info_tensor = self.sim.info_tensor().to_torch()[world_render_idx]
                if info_tensor[agent_idx, -1] == float(
                    gpudrive.EntityType.Padding
                ) or info_tensor[agent_idx, -1] == float(
                    gpudrive.EntityType._None
                ):
                    continue

                self.surf.fill(self.BACKGROUND_COLOR)
                temp_surf = pygame.Surface(
                    (self.surf.get_width(), self.surf.get_height())
                )
                temp_surf.fill(self.BACKGROUND_COLOR)

                agent_info = (
                    self.sim.self_observation_tensor()
                    .to_torch()[world_render_idx, agent_idx, :]
                    .cpu()
                    .detach()
                    .numpy()
                )

                lidar_data = (
                    self.sim.lidar_tensor()
                    .to_torch()[world_render_idx, agent_idx, :, :, :] # shape is (num_worlds, num_agents, num_planes, num_samples, 2)
                    .cpu()
                    .detach()
                    .numpy()
                )
                
                for lidar_plane in lidar_data:
                    self.plotLidar(temp_surf, lidar_plane, world_render_idx)

                goal_pos = agent_info[3:5]  # x, y
                agent_size = agent_info[1:3]  # length, width

                agent_corners = PyGameVisualizer.compute_agent_corners(
                    (0, 0), agent_size[1], agent_size[0], 0
                )
                agent_corners = [
                    self.scale_coords(corner, world_render_idx)
                    for corner in agent_corners
                ]
                current_goal_scaled = self.scale_coords(
                    goal_pos, world_render_idx
                )

                pygame.gfxdraw.aapolygon(
                    temp_surf, agent_corners, self.COLOR_LIST[0]
                )
                pygame.gfxdraw.filled_polygon(
                    temp_surf, agent_corners, self.COLOR_LIST[0]
                )

                self.draw_circle(
                    temp_surf,
                    current_goal_scaled,
                    self.goal_radius * self.zoom_scales_x[world_render_idx],
                    self.COLOR_LIST[0],
                )

                # blit temp surf on self.surf
                self.surf.blit(temp_surf, (0, 0))
                # Capture the RGB array for the agent's view
                render_rgbs.append(
                    PyGameVisualizer._create_image_array(self.surf)
                )
            return render_rgbs

    @staticmethod
    def _create_image_array(surf):
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)
        )

    def destroy(self):
        pygame.display.quit()
        pygame.quit()
