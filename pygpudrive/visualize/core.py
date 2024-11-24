import os
import torch
from pathlib import Path
import mediapy
import matplotlib
from typing import Tuple, Optional, List, Dict, Any, Union

from pygpudrive.visualize.color import (
    ROAD_GRAPH_COLORS,
    ROAD_GRAPH_TYPE_NAMES,
    REL_OBS_OBJ_COLORS,
)
from pygpudrive.visualize import utils


def plot_agent_observation(
    agent_idx: int,
    env_idx: int,
    observation_roadgraph: torch.Tensor = None,
    observation_ego: torch.Tensor = None,
    observation_partner: torch.Tensor = None,
    time_step: int = 0,
    x_lim: Tuple[float, float] = (-100, 100),
    y_lim: Tuple[float, float] = (-100, 100),
    viz_config=None,
):
    """Plot observation from agent POV to inspect the information available to the agent.
    Args:
        agent_idx (int): Index of the agent whose observation is to be plotted.
        env_idx (int): Index of the environment in the batch.
        observation_roadgraph (torch.Tensor, optional): Roadgraph tensor. Defaults to None.
        observation_ego (torch.Tensor, optional): Ego observation tensor. Defaults to None.
        observation_partner (torch.Tensor, optional): Partner observation tensor. Defaults to None.
        time_step (int, optional): Time step of the observation. Defaults to 0.
        x_lim (Tuple[float, float], optional): x-axis limits. Defaults to (-100, 100).
        y_lim (Tuple[float, float], optional): y-axis limits. Defaults to (-100, 100).
        viz_config ([type], optional): Visualization config. Defaults to None.
    """

    # Check if agent index is valid, otherwise return None
    if observation_ego.id[env_idx, agent_idx] == -1:
        return None, None

    viz_config = (
        utils.VizConfig()
        if viz_config is None
        else utils.VizConfig(**viz_config)
    )

    fig, ax = utils.init_fig_ax(viz_config)
    ax.set_title(f"$t$ = {time_step} | obs agent: {agent_idx}")

    # Plot roadgraph if provided
    if observation_roadgraph is not None:
        for road_type, type_name in ROAD_GRAPH_TYPE_NAMES.items():
            mask = (
                observation_roadgraph.type[env_idx, agent_idx, :] == road_type
            )
            ax.scatter(
                observation_roadgraph.x[env_idx, agent_idx, mask],
                observation_roadgraph.y[env_idx, agent_idx, mask],
                c=[ROAD_GRAPH_COLORS[road_type]],
                s=10,
                label=type_name,
            )

    # Plot partner agents if provided
    if observation_partner is not None:
        partner_positions = torch.stack(
            (
                observation_partner.rel_pos_x[env_idx, agent_idx, :, :]
                .squeeze()
                .cpu(),
                observation_partner.rel_pos_y[env_idx, agent_idx, :, :]
                .squeeze()
                .cpu(),
            ),
            dim=1,
        )  # Shape: (num_partners, 2)

        utils.plot_bounding_box(
            ax=ax,
            center=partner_positions,
            vehicle_length=observation_partner.vehicle_length[
                env_idx, agent_idx, :, :
            ].squeeze(),
            vehicle_width=observation_partner.vehicle_width[
                env_idx, agent_idx, :, :
            ].squeeze(),
            orientation=observation_partner.orientation[
                env_idx, agent_idx, :, :
            ].squeeze(),
            color=REL_OBS_OBJ_COLORS["other_agents"],
            label="Other agents",
            alpha=0.8,
        )

    if observation_ego is not None:
        ego_agent_color = (
            "darkred"
            if observation_ego.is_collided[env_idx, agent_idx]
            else REL_OBS_OBJ_COLORS["ego"]
        )
        utils.plot_bounding_box(
            ax=ax,
            center=(0, 0),
            vehicle_length=observation_ego.vehicle_length[
                env_idx, agent_idx
            ].item(),
            vehicle_width=observation_ego.vehicle_width[
                env_idx, agent_idx
            ].item(),
            orientation=0.0,
            color=ego_agent_color,
            alpha=1.0,
            label="Ego agent",
        )

        # Add an arrow for speed
        speed = observation_ego.speed[env_idx, agent_idx].item()
        ax.arrow(
            0,
            0,  # Start at the ego vehicle's position
            speed,
            0,  # Arrow points to the right, proportional to speed
            head_width=0.5,
            head_length=0.7,
            fc=REL_OBS_OBJ_COLORS["ego"],
            ec=REL_OBS_OBJ_COLORS["ego"],
        )

        ax.plot(
            observation_ego.rel_goal_x[env_idx, agent_idx],
            observation_ego.rel_goal_y[env_idx, agent_idx],
            markersize=23,
            label="Goal",
            marker="*",
            markeredgecolor="k",
            linestyle="None",
            color=REL_OBS_OBJ_COLORS["ego_goal"],
        )[0]

    # fig.legend(
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 0.1),
    #     ncol=5,
    #     fontsize=10,
    #     title="Elements",
    # )
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    # ax.set_xticks([])
    # ax.set_yticks([])

    return fig, ax
