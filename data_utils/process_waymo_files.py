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
import argparse
import logging
import psutil
from pathlib import Path
import warnings
from typing import Any, Dict, Optional, List
from pdb import set_trace as T
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2, map_pb2
from datatypes import MapElementIds
import trimesh
from multiprocessing import Pool, cpu_count
import numpy as np

# To filter out warnings before tensorflow is imported
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)


def wrap_yaws(yaws):
    """Wraps yaw angles between pi and -pi radians."""
    return (yaws + np.pi) % (2 * np.pi) - np.pi


ERR_VAL = -1e4

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
        "heading": [ # In radians between [-pi, pi]
            (state.heading + np.pi) % (2 * np.pi) - np.pi if state.valid else ERR_VAL
            for state in states
        ], 
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
    obj["id"] = track.id
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
            {"x": p.x, "y": p.y, "z": p.z}
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


# Meshes for collision checking
def _filter_small_segments(segments, min_length=1e-6):
    """Filter out segments that are too short."""
    valid_segments = []
    for segment in segments:
        start, end = segment
        length = np.linalg.norm(np.array(end) - np.array(start))
        if length >= min_length:
            valid_segments.append(segment)
    return valid_segments


def _generate_mesh(segments, height=2.0, width=0.2):
    segments = np.array(segments, dtype=np.float64)
    starts, ends = segments[:, 0, :], segments[:, 1, :]
    directions = ends - starts
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    unit_directions = directions / lengths

    # Create the base box mesh with the height along the z-axis
    base_box = trimesh.creation.box(extents=[1.0, width, height])
    base_box.apply_translation([0.5, 0, 0])  # Align box's origin to its start
    z_axis = np.array([0, 0, 1])
    angles = np.arctan2(
        unit_directions[:, 1], unit_directions[:, 0]
    )  # Rotation in the XY plane

    rectangles = []
    lengths = lengths.flatten()

    for i, (start, length, angle) in enumerate(zip(starts, lengths, angles)):
        # Copy the base box and scale to match segment length
        scaled_box = base_box.copy()
        scaled_box.apply_scale([length, 1.0, 1.0])

        # Apply rotation around the z-axis
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle, z_axis
        )
        scaled_box.apply_transform(rotation_matrix)

        # Translate the box to the segment's starting point
        scaled_box.apply_translation(start)

        rectangles.append(scaled_box)

    # Concatenate all boxes into a single mesh
    mesh = trimesh.util.concatenate(rectangles)
    return mesh


def _create_agent_box_mesh(position, heading, length, width, height):
    """Create a box mesh for an agent at a given position and orientation.
    
    Args:
        position (list): [x, y, z] position
        heading (float): yaw angle in radians
        length (float): length of the box
        width (float): width of the box 
        height (float): height of the box
        
    Returns:
        trimesh.Trimesh: Box mesh positioned and oriented correctly
    """
    # Create box centered at origin
    box = trimesh.creation.box(extents=[length, width, height])
    
    # Rotate box to align with heading
    z_axis = np.array([0, 0, 1])
    rotation_matrix = trimesh.transformations.rotation_matrix(heading, z_axis)
    box.apply_transform(rotation_matrix)
    
    # Move box to position
    box.apply_translation(position)
    
    return box


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
        # there is a traffic light but we don't want traffic light scenes so just return
        if len(traffic_light_dict) > 0:
            return
        for id, value in traffic_light_dict.items():
            for key in all_keys:
                tl_dict[id][key].append(value[key])
            tl_dict[id]["time_index"].append(i)
        i += 1

    # Construct the map states
    roads = []
    edge_points = []
    edge_segments = []

    for map_feature in protobuf.map_features:
        road = _init_road(map_feature)
        if road is not None:
            roads.append(road)
            if road["type"] == "road_edge":
                # Collect points for 3D structure detection
                edge_vertices = [[r["x"], r["y"], r["z"]] for r in road["geometry"]]
                edge_points.extend(edge_vertices)
                # Collect edge segments for collision checking
                edge_segments.extend([
                    [edge_vertices[i], edge_vertices[i + 1]]
                    for i in range(len(edge_vertices) - 1)
                ])
    
    # Check for 3D structures
    if len(edge_points) > 0:
        edge_points = np.array(edge_points)
        if len(edge_points) > 0:
            # Calculate pairwise distances in xy plane efficiently
            xy_points = edge_points[:, :2]
            # Use broadcasting for memory efficiency
            tolerance = 0.2
            has_3d = False
            
            # Process in chunks to avoid memory issues
            chunk_size = 1000
            for i in range(0, len(xy_points), chunk_size):
                chunk = xy_points[i:i + chunk_size]
                # Calculate distances between current chunk and all points
                dists = np.linalg.norm(chunk[:, np.newaxis] - xy_points, axis=2)
                potential_pairs = np.where((dists < tolerance) & (dists > 0))
                
                # Check z-values for identified pairs
                for p1, p2 in zip(*potential_pairs):
                    p1_idx = i + p1  # Adjust index for chunking
                    if abs(edge_points[p1_idx, 2] - edge_points[p2, 2]) > tolerance:
                        has_3d = True
                        break
                
                if has_3d:
                    break
            
            # Skip this scenario if it has 3D structures
            if has_3d:
                return

    # Construct road edges for collision checking
    edge_segments = _filter_small_segments(edge_segments)
    edge_mesh = _generate_mesh(edge_segments)

    # Create collision managers
    road_collision_manager = trimesh.collision.CollisionManager()
    road_collision_manager.add_object("road_edges", edge_mesh)
    agent_collision_manager = trimesh.collision.CollisionManager()
    trajectory_collision_manager = trimesh.collision.CollisionManager()

    
    # Construct object states
    objects = []
    for track in protobuf.tracks:
        obj = _init_object(track)
        if obj is not None:
            if obj["type"] not in ["vehicle", "cyclist"]:
                obj["mark_as_expert"] = False
                objects.append(obj)
                continue

            # Find first valid position
            first_valid_idx = next((i for i, valid in enumerate(obj["valid"]) if valid), None)
            if first_valid_idx is not None:
                # Create agent at initial position
                initial_pos = [
                    obj["position"][first_valid_idx]["x"],
                    obj["position"][first_valid_idx]["y"],
                    obj["position"][first_valid_idx]["z"]
                ]
                initial_heading = obj["heading"][first_valid_idx]
                initial_box = _create_agent_box_mesh(
                    initial_pos,
                    initial_heading,
                    obj["length"],
                    obj["width"],
                    obj["height"]
                )
                agent_collision_manager.add_object(str(obj["id"]), initial_box)

                # Create trajectory mesh
                if False in obj["valid"]:
                    # Create trajectory segments of only valid positions
                    trajectory_segments = []
                    for i in range(len(obj["position"]) - 1):
                        if obj["valid"][i] and obj["valid"][i + 1]:
                            trajectory_segments.append(
                                [
                                    [
                                        obj["position"][i]["x"],
                                        obj["position"][i]["y"],
                                        obj["position"][i]["z"],
                                    ],
                                    [
                                        obj["position"][i + 1]["x"],
                                        obj["position"][i + 1]["y"],
                                        obj["position"][i + 1]["z"],
                                    ],
                                ]
                            )
                else:
                    obj_vertices = [
                        [pos["x"], pos["y"], pos["z"]] for pos in obj["position"]
                    ]
                    trajectory_segments = [
                        [obj_vertices[i], obj_vertices[i + 1]]
                        for i in range(len(obj_vertices) - 1)
                    ]

                trajectory_segments = _filter_small_segments(trajectory_segments)
                if len(trajectory_segments) > 0:
                    trajectory_mesh = _generate_mesh(trajectory_segments)
                    trajectory_collision_manager.add_object(str(obj["id"]), trajectory_mesh)
                
                objects.append(obj)
    
    # Check collisions between all init agent positions
    _, agent_collision_pairs = agent_collision_manager.in_collision_internal(return_names=True)
    
    # Check collisions between init agent positions and road edges
    _, road_collision_pairs = agent_collision_manager.in_collision_other(
        road_collision_manager, return_names=True
    )

    # Check trajectory collisions with road edges
    _, trajectory_collision_pairs = trajectory_collision_manager.in_collision_other(
        road_collision_manager, return_names=True
    )
    
    # Create sets of colliding agent IDs
    colliding_agents = set()

    # Add agents that collide with each other at first step
    for agent1, agent2 in agent_collision_pairs:
        colliding_agents.add(agent1)
        colliding_agents.add(agent2)
    
    # Add agents that collide with road edges
    road_colliding_agents = set(agent_id for agent_id, _ in road_collision_pairs)
    colliding_agents.update(road_colliding_agents)

    # Add agents whose trajectories collide with road edges
    trajectory_colliding_agents = set(agent_id for agent_id, _ in trajectory_collision_pairs)
    colliding_agents.update(trajectory_colliding_agents)
    
    # Update mark_as_expert based on initial collisions
    for index, obj in enumerate(objects):
        if obj["type"] in ["vehicle", "cyclist"]:
            if str(obj["id"]) in colliding_agents:
                objects[index]["mark_as_expert"] = True
            else:
                objects[index]["mark_as_expert"] = False
                
    # Parse metadata
    sdc_track_index = protobuf.sdc_track_index
    objects_of_interest = list(protobuf.objects_of_interest)
    tracks_to_predict = [
    {
        "track_index": track.track_index,
        "difficulty": track.difficulty
    }
    for track in protobuf.tracks_to_predict
]
    metadata = {
        "sdc_track_index" : sdc_track_index,
        "objects_of_interest" : objects_of_interest,
        "tracks_to_predict" : tracks_to_predict
    }

    scenario_dict = {
        "name": scenario_path.split("/")[-1],
        "scenario_id": scenario_id,
        "objects": objects,
        "roads": roads,
        "tl_states": tl_dict,
        "metadata": metadata
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


def process_scene(args):
    scene_proto, output_dir, file_prefix, scene_count, id_as_filename = args
    try:
        scenario_id = scene_proto.scenario_id
        file_suffix = (
            f"{scenario_id}.json" if id_as_filename else f"{scene_count}.json"
        )
        waymo_to_scenario(
            scenario_path=os.path.join(
                output_dir, f"{file_prefix}{file_suffix}"
            ),
            protobuf=scene_proto,
        )
    except Exception as e:
        logging.error(
            f"Error processing scene {file_prefix}{scene_count}: {e}"
        )


# Scenario-level parallelization
def process_file(args):
    """Process a single TFRecord file."""
    filename, output_dir, id_as_filename, num_workers = args

    # Read the records in batches
    mem_info = psutil.virtual_memory()
    available_memory = mem_info.available / (1024**3)
    usable_memory = int(available_memory * 0.9)
    # 10 scenes take 1 Gb at max
    batch_size = 12 * usable_memory
    tfrecord_dataset = tf.data.TFRecordDataset(filename, compression_type="")
    tf_dataset_iter = as_proto_iterator(tfrecord_dataset)

    scene_count = 0
    file_prefix = f"{str(filename).split('.')[-1]}_"
    scene_batch = []

    for scene_proto in tf_dataset_iter:
        scene_batch.append((scene_proto, scene_count))
        scene_count += 1
        if len(scene_batch) == batch_size:
            # Process the batch
            with Pool(num_workers) as pool:
                pool.map(
                    process_scene,
                    [
                        (
                            scene_proto,
                            output_dir,
                            file_prefix,
                            count,
                            id_as_filename,
                        )
                        for scene_proto, count in scene_batch
                    ],
                )
            scene_batch = []

    # Process any remaining scenes
    if scene_batch:
        with Pool(num_workers) as pool:
            pool.map(
                process_scene,
                [
                    (
                        scene_proto,
                        output_dir,
                        file_prefix,
                        count,
                        id_as_filename,
                    )
                    for scene_proto, count in scene_batch
                ],
            )


def process_data(args):

    if args.dataset == "all":
        datasets = ["training", "validation", "testing"]
    elif args.dataset == "training":
        datasets = ["training"]
    elif args.dataset == "validation":
        datasets = ["validation"]
    elif args.dataset == "testing":
        datasets = ["testing"]
    else:
        raise ValueError(
            "Invalid dataset name. Must be one of: 'all', 'training', 'validation', or 'testing'"
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
        # Process the files one at a time
        for filename in tqdm(filenames, unit="file"):
            process_file(
                (
                    str(filename),
                    output_dir,
                    args.id_as_filename,
                    args.num_workers,
                )
            )
        logging.info("Done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert TFRecord files to JSON. \
            Note: This takes about 45 seconds per tfrecord file (=500 traffic scenes)."
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
        default=False,
        action="store_true",
        help="Use the unique scenario id as the filename",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count(),
        help="Number of worker processes to use",
    )

    args = parser.parse_args()

    process_data(args)
