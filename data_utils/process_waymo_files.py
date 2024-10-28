"""
Convert Waymo Open Dataset TFRecord files to JSON format.
See https://waymo.com/open/data/motion/tfexample for the tfrecord structure; and
https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto
https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/scenario.proto
for the proto structure.
"""
from collections import defaultdict
import os
import json
import math
import argparse
import logging
from pathlib import Path
import warnings
from typing import Any, Dict, Optional
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2, map_pb2

from data_utils.datatypes import MapElementIds

# To filter out warnings before tensorflow is imported
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)

ERR_VAL = 1e-4

_WAYMO_OBJECT_STR = {
    scenario_pb2.Track.TYPE_UNSET: "unset",
    scenario_pb2.Track.TYPE_VEHICLE: "vehicle",
    scenario_pb2.Track.TYPE_PEDESTRIAN: "pedestrian",
    scenario_pb2.Track.TYPE_CYCLIST: "cyclist",
    scenario_pb2.Track.TYPE_OTHER: "other",
}

_WAYMO_ROAD_STR = {
    map_pb2.TrafficSignalLaneState.LANE_STATE_UNKNOWN: "unknown",
    map_pb2.TrafficSignalLaneState.LANE_STATE_ARROW_STOP: "arrow_stop",
    map_pb2.TrafficSignalLaneState.LANE_STATE_ARROW_CAUTION: "arrow_caution",
    map_pb2.TrafficSignalLaneState.LANE_STATE_ARROW_GO: "arrow_go",
    map_pb2.TrafficSignalLaneState.LANE_STATE_STOP: "stop",
    map_pb2.TrafficSignalLaneState.LANE_STATE_CAUTION: "caution",
    map_pb2.TrafficSignalLaneState.LANE_STATE_GO: "go",
    map_pb2.TrafficSignalLaneState.LANE_STATE_FLASHING_STOP: "flashing_stop",
    map_pb2.TrafficSignalLaneState.LANE_STATE_FLASHING_CAUTION: "flashing_caution",
}

_WAYMO_LANE_TYPES = {
    map_pb2.LaneCenter.TYPE_UNDEFINED: MapElementIds.LANE_UNDEFINED,
    map_pb2.LaneCenter.TYPE_FREEWAY: MapElementIds.LANE_FREEWAY,
    map_pb2.LaneCenter.TYPE_SURFACE_STREET: MapElementIds.LANE_SURFACE_STREET,
    map_pb2.LaneCenter.TYPE_BIKE_LANE: MapElementIds.LANE_BIKE_LANE,
}

_WAYMO_ROAD_LINE_TYPES = {
    map_pb2.RoadLine.TYPE_UNKNOWN: MapElementIds.ROAD_LINE_UNKNOWN,
    map_pb2.RoadLine.TYPE_BROKEN_SINGLE_WHITE: MapElementIds.ROAD_LINE_BROKEN_SINGLE_WHITE,
    map_pb2.RoadLine.TYPE_SOLID_SINGLE_WHITE: MapElementIds.ROAD_LINE_SOLID_SINGLE_WHITE,
    map_pb2.RoadLine.TYPE_SOLID_DOUBLE_WHITE: MapElementIds.ROAD_LINE_SOLID_DOUBLE_WHITE,
    map_pb2.RoadLine.TYPE_BROKEN_SINGLE_YELLOW: MapElementIds.ROAD_LINE_BROKEN_SINGLE_YELLOW,
    map_pb2.RoadLine.TYPE_BROKEN_DOUBLE_YELLOW: MapElementIds.ROAD_LINE_BROKEN_DOUBLE_YELLOW,
    map_pb2.RoadLine.TYPE_SOLID_SINGLE_YELLOW: MapElementIds.ROAD_LINE_SOLID_SINGLE_YELLOW,
    map_pb2.RoadLine.TYPE_SOLID_DOUBLE_YELLOW: MapElementIds.ROAD_LINE_SOLID_DOUBLE_YELLOW,
    map_pb2.RoadLine.TYPE_PASSING_DOUBLE_YELLOW: MapElementIds.ROAD_LINE_PASSING_DOUBLE_YELLOW,
}

_WAYMO_ROAD_EDGE_TYPES = {
    map_pb2.RoadEdge.TYPE_UNKNOWN: MapElementIds.ROAD_EDGE_UNKNOWN,
    map_pb2.RoadEdge.TYPE_ROAD_EDGE_BOUNDARY: MapElementIds.ROAD_EDGE_BOUNDARY,
    map_pb2.RoadEdge.TYPE_ROAD_EDGE_MEDIAN: MapElementIds.ROAD_EDGE_MEDIAN,
}


def feature_class_to_map_id(map_feature):
    """
    Converts the map feature types defined in the proto to the ones
    defined in the datatypes.py, to ensure consistency with Waymax.
    """
    if map_feature.HasField("lane"):
        map_element_id = _WAYMO_LANE_TYPES.get(map_feature.lane.type)
    elif map_feature.HasField("road_line"):
        map_element_id = _WAYMO_ROAD_LINE_TYPES.get(map_feature.road_line.type)
    elif map_feature.HasField("road_edge"):
        map_element_id = _WAYMO_ROAD_EDGE_TYPES.get(map_feature.road_edge.type)
    elif map_feature.HasField("stop_sign"):
        map_element_id = MapElementIds.STOP_SIGN
    elif map_feature.HasField("crosswalk"):
        map_element_id = MapElementIds.CROSSWALK
    elif map_feature.HasField("speed_bump"):
        map_element_id = MapElementIds.SPEED_BUMP
    # New in WOMD v1.2.0: Driveway entrances
    elif map_feature.HasField("driveway"):
        map_element_id = MapElementIds.DRIVEWAY
    else:
        map_element_id = MapElementIds.UNKNOWN

    return int(map_element_id)


def _parse_object_state(
    states: scenario_pb2.ObjectState, final_state: scenario_pb2.ObjectState
) -> Dict[str, Any]:
    """Construct a dict representing the trajectory and goals of an object.

    Args:
        states (scenario_pb2.ObjectState): Protobuf of object state
        final_state (scenario_pb2.ObjectState): Protobuf of last valid object state.

    Returns
    -------
        Dict[str, Any]: Dict representing an object.
    """
    return {
        "position": [
            {"x": state.center_x, "y": state.center_y, "z": state.center_z}
            if state.valid
            else {"x": ERR_VAL, "y": ERR_VAL, "z": ERR_VAL}
            for state in states
        ],
        "width": final_state.width,
        "length": final_state.length,
        "height": final_state.height,
        "heading": [
            math.degrees(state.heading) if state.valid else ERR_VAL
            for state in states
        ],  # Use rad here?
        "velocity": [
            {"x": state.velocity_x, "y": state.velocity_y}
            if state.valid
            else {"x": ERR_VAL, "y": ERR_VAL}
            for state in states
        ],
        "valid": [state.valid for state in states],
        "goalPosition": {
            "x": final_state.center_x,
            "y": final_state.center_y,
            "z": final_state.center_z,
        },
    }


def _init_tl_object(mapstate: scenario_pb2.DynamicMapState) -> Dict[int, Any]:
    """Construct a dict representing the traffic light states.

    Args:
        mapstate (scenario_pb2.DynamicMapState) : protobuf of map state (traffic lights)

    Returns:
        Dict[int, Any] : Dict representing map state
    """
    returned_dict = {}
    for lane_state in mapstate.lane_states:
        returned_dict[lane_state.lane] = {
            "state": _WAYMO_ROAD_STR[lane_state.state],
            "x": lane_state.stop_point.x,
            "y": lane_state.stop_point.y,
            "z": lane_state.stop_point.z,
        }
    return returned_dict


def _init_object(track: scenario_pb2.Track) -> Optional[Dict[str, Any]]:
    """Construct a dict representing the state of the object (vehicle, cyclist, pedestrian).

    Args:
        track (scenario_pb2.Track): protobuf representing the scenario

    Returns
    -------
        Optional[Dict[str, Any]]: dict representing the trajectory and velocity of an object.
    """
    final_valid_index = 0
    for i, state in enumerate(track.states):
        if state.valid:
            final_valid_index = i

    obj = _parse_object_state(track.states, track.states[final_valid_index])
    obj["type"] = _WAYMO_OBJECT_STR[track.object_type]
    return obj


def _init_road(map_feature: map_pb2.MapFeature) -> Optional[Dict[str, Any]]:
    """Convert an element of the map protobuf to a dict representing its coordinates and type."""
    feature = map_feature.WhichOneof("feature_data")
    if feature == "stop_sign":
        p = getattr(
            map_feature, map_feature.WhichOneof("feature_data")
        ).position
        geometry = [{"x": p.x, "y": p.y, "z": p.z}]
    elif (
        feature != "crosswalk"
        and feature != "speed_bump"
        and feature != "driveway"
    ):  # For road points
        geometry = [
            {"x": p.x, "y": p.y, "z": p.z, "id": map_feature.id}
            for p in getattr(
                map_feature, map_feature.WhichOneof("feature_data")
            ).polyline
        ]
    else:
        geometry = [
            {"x": p.x, "y": p.y, "z": p.z}
            for p in getattr(
                map_feature, map_feature.WhichOneof("feature_data")
            ).polygon
        ]
    return {
        "geometry": geometry,
        "type": map_feature.WhichOneof("feature_data"),
        "map_element_id": feature_class_to_map_id(map_feature),
        "id": map_feature.id,
    }


def waymo_to_scenario(
    scenario_path: str, protobuf: scenario_pb2.Scenario
) -> None:
    """Dump a JSON File containing the protobuf parsed into the right format.
    See https://waymo.com/open/data/motion/tfexample for the tfrecord structure.

    Args
    ----
        scenario_path (str): path to dump the json file
        protobuf (scenario_pb2.Scenario): the protobuf we are converting
        no_tl (bool, optional): If true, environments with traffic lights are not dumped.
    """
    # read the protobuf file to get the right state
    # write the json file
    # construct the road geometries
    # place the initial position of the vehicles

    # Get unique ID string for a scenario
    scenario_id = protobuf.scenario_id

    # Construct the traffic light states
    tl_dict = defaultdict(
        lambda: {"state": [], "x": [], "y": [], "z": [], "time_index": []}
    )
    all_keys = ["state", "x", "y", "z"]
    i = 0
    for dynamic_map_state in protobuf.dynamic_map_states:
        traffic_light_dict = _init_tl_object(dynamic_map_state)
        for id, value in traffic_light_dict.items():
            for key in all_keys:
                tl_dict[id][key].append(value[key])
            tl_dict[id]["time_index"].append(i)
        i += 1

    # Construct the object states
    objects = []
    for track in protobuf.tracks:
        obj = _init_object(track)
        if obj is not None:
            objects.append(obj)

    # Construct the map states
    roads = []
    for map_feature in protobuf.map_features:
        road = _init_road(map_feature)
        if road is not None:
            roads.append(road)

    scenario_dict = {
        "name": scenario_path.split("/")[-1],
        "scenario_id": scenario_id,
        "objects": objects,
        "roads": roads,
        "tl_states": tl_dict,
    }

    with open(scenario_path, "w") as f:
        json.dump(scenario_dict, f)


def as_proto_iterator(tf_dataset):
    """Parse the tfrecord dataset into a protobuf format."""
    for tfrecord in tf_dataset:
        # Parse the scenario protobuf
        scene_proto = scenario_pb2.Scenario()
        scene_proto.ParseFromString(bytes(tfrecord.numpy()))
        yield scene_proto


def process_data(args):

    if args.dataset == "all":
        datasets = ["training", "validation", "testing"]
    elif args.dataset == "train":
        datasets = ["training"]
    elif args.dataset == "validation":
        datasets = ["validation"]
    elif args.dataset == "testing":
        datasets = ["testing"]
    else:
        raise ValueError(
            "Invalid dataset name. Must be one of: 'all', 'train', 'validation', or 'testing'"
        )

    for dataset in datasets:

        input_dir = os.path.join(args.tfrecord_dir, dataset)
        output_dir = os.path.join(args.output_dir, dataset)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filenames = [
            p for p in Path(input_dir).iterdir() if "tfrecord" in p.suffix
        ]

        assert len(filenames) > 0, f"No TFRecords found in {input_dir}"

        logging.info(
            f"Processing {dataset} data. Found {len(filenames)} files. \n \n"
        )

        # Process the data
        for filename in tqdm(
            filenames,
            total=len(filenames),
            desc="Processing Waymo files",
            colour="green",
        ):
            scene_count = 0
            file_prefix = f"{str(filename).split('.')[-1]}_"

            tfrecord_dataset = tf.data.TFRecordDataset(
                filename,
                compression_type="",
            )
            tf_dataset_iter = as_proto_iterator(tfrecord_dataset)

            for scene_proto in tf_dataset_iter:
                try:
                    if args.id_as_filename:
                        file_suffix = f"{str(scene_proto.scenario_id)}.json"
                    else:
                        file_suffix = f"{scene_count}.json"

                    waymo_to_scenario(
                        scenario_path=os.path.join(
                            output_dir, f"{file_prefix}{file_suffix}"
                        ),
                        protobuf=scene_proto,
                    )

                    scene_count += 1

                except Exception as e:
                    logging.error(
                        f"Error processing record {scene_count}: {e}"
                    )

        logging.info("Done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert TFRecord files to JSON. \
            Note: This takes about 45 seconds per tfrecord file (=50 traffic scenes)."
    )
    parser.add_argument(
        "tfrecord_dir", help="Path to the directory containing TFRecord files"
    )
    parser.add_argument(
        "output_dir",
        help="Directory where JSON files will be saved",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset to process: training, validation, testing, or all",
    )
    parser.add_argument(
        "--id_as_filename",
        default=True,
        help="Use the unique scenario id as the filename",
    )

    args = parser.parse_args()

    process_data(args)
