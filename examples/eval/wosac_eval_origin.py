import itertools
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List

import tensorflow as tf
import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wosac_metrics
from waymo_open_dataset.utils.sim_agents import submission_specs

# import eval.wosac_metrics.metrics as wosac_metrics

from google.protobuf import text_format
from torch import Tensor, tensor
from torchmetrics import Metric
from waymo_open_dataset.protos import (
    scenario_pb2,
    sim_agents_metrics_pb2,
    sim_agents_submission_pb2,
)


class WOSACMetrics(Metric):
    """
    validation metrics based on ground truth trajectory, using waymo_open_dataset api
    """

    def __init__(self, challenge_type = None, prefix: str = "", ego_only: bool = False) -> None:
        super().__init__()
        self.is_mp_init = False
        self.prefix = prefix
        self.ego_only = ego_only
        if challenge_type is None:
            self.challenge_type = submission_specs.ChallengeType.SIM_AGENTS
        else:
            self.challenge_type = challenge_type

        self.wosac_config = load_metrics_config(self.challenge_type)

        self.field_names = [
            "metametric",
            "average_displacement_error",
            "linear_speed_likelihood",
            "linear_acceleration_likelihood",
            "angular_speed_likelihood",
            "angular_acceleration_likelihood",
            "distance_to_nearest_object_likelihood",
            "collision_indication_likelihood",
            "time_to_collision_likelihood",
            "distance_to_road_edge_likelihood",
            "offroad_indication_likelihood",
            "min_average_displacement_error",
            "traffic_light_violation_likelihood",
            "simulated_collision_rate",
            "simulated_offroad_rate",
        ]
        for k in self.field_names:
            self.add_state(k, default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("scenario_counter", default=tensor(0.0), dist_reduce_fx="sum")
        tf.config.set_visible_devices([], "GPU")

    @staticmethod
    def _compute_scenario_metrics(
        config, scenario_file, scenario_rollout, ego_only, scenario_rollouts_mask=None
    ) -> sim_agents_metrics_pb2.SimAgentMetrics:
        scenario = scenario_pb2.Scenario()
        for data in tf.data.TFRecordDataset([scenario_file], compression_type=""):
            scenario.ParseFromString(bytes(data.numpy()))
            break
        if ego_only:
            for i in range(len(scenario.tracks)):
                if i != scenario.sdc_track_index:
                    for t in range(91):
                        scenario.tracks[i].states[t].valid = False
            while len(scenario.tracks_to_predict) > 1:
                scenario.tracks_to_predict.pop()
            scenario.tracks_to_predict[0].track_index = scenario.sdc_track_index

        return wosac_metrics.compute_scenario_metrics_for_bundle(
            config, scenario, scenario_rollout,# scenario_rollouts_mask=scenario_rollouts_mask
        )

    def update(
        self,
        scenario_files: List[str],
        scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts],
        scenario_rollout_masks = None,
    ) -> None:

        if scenario_rollout_masks is None:
            scenario_rollout_masks = [None] * len(scenario_rollouts)
        pool_scenario_metrics = []
        for _scenario, _scenario_rollout, _scenario_mask in zip(scenario_files, scenario_rollouts, scenario_rollout_masks):
            try:
                pool_scenario_metrics.append(
                    self._compute_scenario_metrics(
                        self.wosac_config, _scenario, _scenario_rollout, self.ego_only,
                        # scenario_rollouts_mask=_scenario_mask
                    )
                )
            except Exception as e:
                print(f"Error processing scenario {_scenario_rollout.scenario_id}")
                print(e)

        for scenario_metrics in pool_scenario_metrics:
            self.scenario_counter += 1
            self.metametric += scenario_metrics.metametric
            self.average_displacement_error += (
                scenario_metrics.average_displacement_error
            )
            self.linear_speed_likelihood += scenario_metrics.linear_speed_likelihood
            self.linear_acceleration_likelihood += (
                scenario_metrics.linear_acceleration_likelihood
            )
            self.angular_speed_likelihood += scenario_metrics.angular_speed_likelihood
            self.angular_acceleration_likelihood += (
                scenario_metrics.angular_acceleration_likelihood
            )
            self.distance_to_nearest_object_likelihood += (
                scenario_metrics.distance_to_nearest_object_likelihood
            )
            self.collision_indication_likelihood += (
                scenario_metrics.collision_indication_likelihood
            )
            self.time_to_collision_likelihood += (
                scenario_metrics.time_to_collision_likelihood
            )
            self.distance_to_road_edge_likelihood += (
                scenario_metrics.distance_to_road_edge_likelihood
            )
            self.offroad_indication_likelihood += (
                scenario_metrics.offroad_indication_likelihood
            )
            self.min_average_displacement_error += (
                scenario_metrics.min_average_displacement_error
            )
            self.traffic_light_violation_likelihood += (
                scenario_metrics.traffic_light_violation_likelihood
            )
            self.simulated_collision_rate += scenario_metrics.simulated_collision_rate
            self.simulated_offroad_rate += scenario_metrics.simulated_offroad_rate

    def compute(self) -> Dict[str, Tensor]:
        metrics_dict = {}
        for k in self.field_names:
            metrics_dict[k] = getattr(self, k) / self.scenario_counter

        mean_metrics = sim_agents_metrics_pb2.SimAgentMetrics(
            scenario_id="", **metrics_dict
        )
        final_metrics = wosac_metrics.aggregate_metrics_to_buckets(
            self.wosac_config, mean_metrics
        )

        out_dict = {
            f"{self.prefix}/wosac/realism_meta_metric": final_metrics.realism_meta_metric,
            f"{self.prefix}/wosac/kinematic_metrics": final_metrics.kinematic_metrics,
            f"{self.prefix}/wosac/interactive_metrics": final_metrics.interactive_metrics,
            f"{self.prefix}/wosac/map_based_metrics": final_metrics.map_based_metrics,
            f"{self.prefix}/wosac/min_ade": final_metrics.min_ade,
            f"{self.prefix}/wosac/scenario_counter": self.scenario_counter,
        }
        for k in self.field_names:
            out_dict[f"{self.prefix}/wosac_likelihood/{k}"] = metrics_dict[k]

        return out_dict


def load_metrics_config(
    challenge_type: submission_specs.ChallengeType,
) -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
    """Loads the `SimAgentMetricsConfig` used for the challenge."""
    import waymo_open_dataset
    pyglib_resource = waymo_open_dataset.__path__[0]
    if challenge_type == submission_specs.ChallengeType.SIM_AGENTS:
        config_path = f'{pyglib_resource}/wdl_limited/sim_agents_metrics/challenge_2025_sim_agents_config.textproto' # pylint: disable=line-too-long
    elif challenge_type == submission_specs.ChallengeType.SCENARIO_GEN:
        config_path = f'{pyglib_resource}/wdl_limited/sim_agents_metrics/challenge_2025_scenario_gen_config.textproto'  # pylint: disable=line-too-long
    else:
        raise ValueError(f'Unsupported {challenge_type=}')
    with open(config_path, 'r') as f:
        config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
        text_format.Parse(f.read(), config)
    return config