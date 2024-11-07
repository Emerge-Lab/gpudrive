import torch
import numpy as np
import jax
import sys

sys.path.append("/home/zhiyu/Projects/Project12/stage9")

from jax import numpy as jnp
from waymax import datatypes
from waymax.agents import actor_core

import matplotlib as mpl
from matplotlib import pyplot as plt
from integrations.models.vbd.waymax_visualization import utils
from integrations.models.vbd.waymax_visualization import color
from integrations.models.vbd.waymax_visualization import viz as visualization
from waymax import config as _config
from typing import Any, Optional


# Parameters
MAX_NUM_OBJECTS = 32
MAX_MAP_POINTS = 3000
MAX_POLYLINES = 256
MAX_TRAFFIC_LIGHTS = 16
num_points_polyline = 30


# Helper functions
def plot_simulator_state(
    state: datatypes.SimulatorState,
    ego_goal: np.array = None,
    adv_goal: np.array = None,
    chosen_goals: np.array = None,
    ego_idx: int = -1,
    adv_idx: int = -1,
    goal_scores: np.array = None,
    use_log_traj: bool = True,
    viz_config: Optional[dict[str, Any]] = None,
    highlight_obj: _config.ObjectType = _config.ObjectType.SDC,
    show_goals: bool = True,
) -> np.ndarray:
    """Plots np array image for SimulatorState.

    Args:
        state: A SimulatorState instance.
        use_log_traj: Set True to use logged trajectory, o/w uses simulated
        trajectory.
        viz_config: dict for optional config.
        batch_idx: optional batch index.
        highlight_obj: Represents the type of objects that will be highlighted with
        `color.COLOR_DICT['controlled']` color.

    Returns:
        np image.
    """

    if state.shape:
        raise ValueError(
            "Expecting 0 batch dimension, got %s" % len(state.shape)
        )

    viz_config = (
        utils.VizConfig()
        if viz_config is None
        else utils.VizConfig(**viz_config)
    )
    fig, ax = utils.init_fig_ax(viz_config)

    # 1. Plots trajectory.
    traj = state.log_trajectory if use_log_traj else state.sim_trajectory
    indices = np.arange(traj.num_objects) if viz_config.show_agent_id else None
    is_controlled = datatypes.get_control_mask(
        state.object_metadata, highlight_obj
    )
    visualization.plot_trajectory(
        ax,
        traj,
        is_controlled,
        time_idx=state.timestep,
        indices=indices,
        is_ego=ego_idx,
        is_adv=adv_idx,
    )  # pytype: disable=wrong-arg-types  # jax-ndarray

    # 2. Plots road graph elements.
    visualization.plot_roadgraph_points(
        ax, state.roadgraph_points, verbose=False
    )
    visualization.plot_traffic_light_signals_as_points(
        ax, state.log_traffic_light, state.timestep, verbose=False
    )

    # ax.scatter(6230, 9530, color='green', marker='x', s=50)
    # ax.scatter(6258, 9500, color='red', marker='x', s=50)

    # 3. Plots goals for ego and adv
    if show_goals:
        i = 0
        ts = 40

        for goal in ego_goal:
            alpha = goal_scores[0, i].item()
            ax.plot(
                goal[ts, 0],
                goal[ts, 1],
                marker="^",
                color=color.COLOR_DICT["ego"],
                ms=6,
                alpha=alpha,
            )
            i = i + 1

        ego_goal = chosen_goals[0]
        ax.plot(
            ego_goal[ts, 0],
            ego_goal[ts, 1],
            marker="X",
            color=color.COLOR_DICT["ego"],
            ms=6,
            alpha=1.0,
        )

        if adv_idx is not None:
            i = 0
            for goal in adv_goal:
                alpha = goal_scores[1, i].item()
                alpha = np.clip(alpha, 0.0, 0.1)
                ax.plot(
                    goal[ts, 0],
                    goal[ts, 1],
                    marker="^",
                    color=color.COLOR_DICT["adv"],
                    ms=6,
                    alpha=alpha,
                )
                i = i + 1

            adv_goal = chosen_goals[1]
            ax.plot(
                adv_goal[ts, 0],
                adv_goal[ts, 1],
                marker="X",
                color=color.COLOR_DICT["adv"],
                ms=6,
                alpha=1.0,
            )

    # 4. Get numpy image, centered on selected agent's current location.
    # [A, 2]
    current_xy = traj.xy[:, state.timestep, :]
    if viz_config.center_agent_idx == -1:
        xy = current_xy[state.object_metadata.is_sdc]
    else:
        xy = current_xy[viz_config.center_agent_idx]

    origin_x, origin_y = xy[0, :2]
    ax.axis(
        (
            origin_x - viz_config.back_x,
            origin_x + viz_config.front_x,
            origin_y - viz_config.back_y,
            origin_y + viz_config.front_y,
        )
    )

    # plots timestep
    t = (state.timestep - 10) / 10
    tt = ax.text(
        origin_x - 0.9 * 50,
        origin_y + 0.9 * 48,
        f"t={t:.1f} s",
        fontsize=18,
        zorder=100,
    )
    tt.set_bbox(dict(facecolor="white", alpha=1, edgecolor="white"))

    # tick off
    plt.tick_params(
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
        bottom=False,
    )

    return utils.img_from_fig(fig)


def sample_to_action(
    sample: np.ndarray,
    is_controlled: jax.Array,
    agents_id: list = None,
    max_num_objects: int = 128,
) -> datatypes.Action:

    """Converts to waymax action."""
    action_dim = sample.shape[-1]
    actions_array = np.zeros((max_num_objects, action_dim))

    if agents_id is None:
        agents_id_full = np.arange(is_controlled.shape[0])
    elif len(agents_id) == is_controlled.shape[0]:
        agents_id_full = agents_id
    elif len(agents_id) < is_controlled.shape[0]:
        agents_id_full = np.full(is_controlled.shape[0], -1, dtype=int)
        agents_id_full[is_controlled] = agents_id
    else:
        raise ValueError("Invalid agents_id size")

    for i, id in enumerate(agents_id_full):
        if id >= 0:
            actions_array[id] = sample[i]

    actions_valid = np.zeros((max_num_objects, 1), dtype=bool)
    actions_valid[agents_id] = True
    actions_valid = jnp.asarray(actions_valid)

    actions = datatypes.Action(
        data=jnp.asarray(actions_array), valid=actions_valid
    )

    return actor_core.WaymaxActorOutput(
        action=actions,
        actor_state=None,
        is_controlled=actions_valid.squeeze(-1),
    )


# def sample_to_action(
#     sample: np.ndarray,
#     is_controlled: jax.Array,
#     agents_id: list = None,
#     max_num_objects: int=128,
# ) -> datatypes.Action:
#     """Converts to waymax action."""
#     action_dim = sample.shape[-1]
#     actions_valid = np.zeros((max_num_objects, 1), dtype=bool)
#     actions_array = np.zeros((max_num_objects, action_dim))

#     # if agents_id is not None:
#     #     actions_valid = np.put(actions_valid, agents_id, True)
#     #     actions_array = np.put(actions_array, agents_id, sample[is_controlled])
#     # else:
#     actions_valid[:is_controlled.shape[0]] = is_controlled[..., None]
#     actions_array[:sample.shape[0]] = sample

#     actions_valid = jnp.asarray(actions_valid)


#     actions = datatypes.Action(data=jnp.asarray(actions_array), valid=actions_valid)

#     return actor_core.WaymaxActorOutput(
#         action=actions,
#         actor_state=None,
#         is_controlled=actions_valid.squeeze(-1),
#     )


def duplicate_batch(batch: dict, num_samples: int):
    """Duplicates the batch for the given number of samples."""
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert value.shape[0] == 1, "Only support batch size of 1"
            batch[key] = torch.cat([value] * num_samples, dim=0)

    return batch


def torch_dict_to_numpy(input: dict):
    output = {}
    for key, value in input.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.detach().cpu().numpy()
        else:
            output[key] = value
    return output


def stack_dict(input: list):
    list_len = len(input)
    if list_len == 0:
        return {}
    key_to_list = {}
    for key in input[0].keys():
        key_to_list[key] = [input[i][key] for i in range(list_len)]

    output = {}
    for key, value in key_to_list.items():
        if isinstance(value[0], np.ndarray):
            output[key] = np.stack(value, axis=0)
        elif isinstance(value[0], dict):
            output[key] = stack_dict(value)
        else:
            output[key] = value

    return output
