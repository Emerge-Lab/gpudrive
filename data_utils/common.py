import waymax
import numpy as np
import mediapy
from pathlib import Path
from tqdm import tqdm
import dataclasses
import pickle
import tensorflow as tf

from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import visualization


def get_common_files_across_simulators(
    path_to_gpudrive_files, waymax_scene_id_iter, save_common_file=True
):
    """Find and store common scenarios in Waymax <> GPUDrive."""

    # Get filenames
    filenames = [p.name for p in Path(path_to_gpudrive_files).iterdir()]
    scenario_hex_ids = [p.split("_")[-1].split(".")[0] for p in filenames]

    # Get the unique ID string for a scenario
    for scenario_id, scenario in waymax_scene_id_iter:
        # Decode bytes and return the scenario ID
        scenario_id = scenario_id.tobytes().decode("utf-8")

        if scenario_id in scenario_hex_ids:
            print(f"Found scenario: {scenario_id}! \n")

            if save_common_file:
                scenario_path = f"waymax_scenario_{scenario_id}.pkl"
                with open(scenario_path, "wb") as f:
                    pickle.dump(scenario, f)
                    print(
                        f"Saved scenario {scenario_id} to {scenario_path} \n"
                    )
                break
        else:
            print(f"Scenario {scenario_id} not found in GPUDrive files.")

    else:
        print("Done")


if __name__ == "__main__":
    data_config = dataclasses.replace(
        _config.WOD_1_1_0_VALIDATION, max_num_objects=32
    )

    data_iter = dataloader.simulator_state_generator(config=data_config)

    # Note: This takes about a minute
    scenario = next(data_iter)

    def _preprocess(serialized: bytes) -> dict[str, tf.Tensor]:
        womd_features = dataloader.womd_utils.get_features_description(
            include_sdc_paths=data_config.include_sdc_paths,
            max_num_rg_points=data_config.max_num_rg_points,
            num_paths=data_config.num_paths,
            num_points_per_path=data_config.num_points_per_path,
        )
        womd_features["scenario/id"] = tf.io.FixedLenFeature([1], tf.string)

        deserialized = tf.io.parse_example(serialized, womd_features)
        parsed_id = deserialized.pop("scenario/id")
        deserialized["scenario/id"] = tf.io.decode_raw(parsed_id, tf.uint8)

        return dataloader.preprocess_womd_example(
            deserialized,
            aggregate_timesteps=data_config.aggregate_timesteps,
            max_num_objects=data_config.max_num_objects,
        )

    def _postprocess(example: dict[str, tf.Tensor]):
        scenario = dataloader.simulator_state_from_womd_dict(example)
        scenario_id = example["scenario/id"]
        return scenario_id, scenario

    scene_id_iter = dataloader.get_data_generator(
        data_config, _preprocess, _postprocess
    )

    get_common_files_across_simulators(
        path_to_gpudrive_files="data/processed/validation",
        waymax_scene_id_iter=scene_id_iter,
        save_common_file=True,
    )
