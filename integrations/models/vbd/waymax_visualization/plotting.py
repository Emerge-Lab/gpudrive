from integrations.models.vbd import waymax_visualization as visualization
import mediapy
import numpy as np
from matplotlib import pyplot as plt


def plot_state(
    current_state,
    log_traj=False,
    traj_preds=None,
    traj_pred_score=None,
    dx=75,
    center_agent_idx=-1,
    filename=None,
    t=None,
    tick_off=False,
    return_ax=False,
    img_size=(400, 400),
):
    viz_config = visualization.utils.VizConfig()
    fig, ax = visualization.utils.init_fig_ax(viz_config)
    if log_traj:
        traj = current_state.log_trajectory
    else:
        traj = current_state.sim_trajectory
    indices = np.arange(traj.num_objects)
    is_controlled = current_state.object_metadata.is_controlled

    visualization.plot_trajectory(
        ax,
        traj,
        is_controlled,
        time_idx=current_state.timestep,
        indices=indices,
    )  # pytype: disable=wrong-arg-types  # jax-ndarray

    # 2. Plots road graph elements.
    visualization.plot_roadgraph_points(
        ax, current_state.roadgraph_points, verbose=False
    )
    visualization.plot_traffic_light_signals_as_points(
        ax,
        current_state.log_traffic_light,
        current_state.timestep,
        verbose=False,
    )

    current_xy = traj.xy[:, current_state.timestep, :]
    if center_agent_idx == -1:
        xy = current_xy[current_state.object_metadata.is_sdc]
        origin_x, origin_y = xy[0, :2]
    else:
        xy = current_xy[center_agent_idx]
        origin_x, origin_y = xy[:2]
    # Zoom

    ax.axis(
        (
            origin_x - dx,
            origin_x + dx,
            origin_y - dx,
            origin_y + dx,
        )
    )
    if t is None:
        t = (current_state.timestep - 10) / 10
    ax.text(
        origin_x - 0.9 * dx, origin_y + 0.9 * dx, f"t={t:.1f} s", fontsize=12
    )

    if tick_off:
        plt.tick_params(
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
            bottom=False,
        )

    if traj_preds is not None and traj_pred_score is not None:
        for traj, score in zip(
            traj_preds.reshape(-1, 80, 5), traj_pred_score.reshape(-1)
        ):
            if score < 0.01:
                continue
            ax.plot(traj[:, 0], traj[:, 1], color="r", alpha=score * 0.8 + 0.2)

    fig.subplots_adjust(
        left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0
    )
    if filename is not None:
        plt.savefig(
            filename, bbox_inches="tight", transparent=False, pad_inches=0.02
        )
    if return_ax:
        return fig, ax
    return mediapy.resize_image(
        visualization.utils.img_from_fig(fig), img_size
    )
