#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPUDrive Scenario Loading and Structure

This script demonstrates how to load and analyze traffic scenarios from the 
Waymo Open Motion Dataset (WOMD) for use with the GPUDrive multi-agent driving simulator.

The script explores the structure of traffic scenarios, including:
- Setting up a data loader
- Examining road objects (vehicles, cyclists)
- Exploring road points (lanes, edges, signs)
- Visualizing scenario data

For more information:
- Waymo Open Dataset: https://github.com/waymo-research/waymo-open-dataset
- Data format: https://waymo.com/open/data/motion/tfexample
- GPUDrive data utilities: https://github.com/Emerge-Lab/gpudrive/tree/main/data_utils
"""

# Dependencies
import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
import seaborn as sns
import pandas as pd

# Set up working directory to find the base 'gpudrive' directory
working_dir = Path.cwd()
while working_dir.name != 'gpudrive':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)

# Configure plotting settings
cmap = ["r", "g", "b", "y", "c"]
plt.rcParams["figure.figsize"] = (8, 3)
sns.set("notebook", font_scale=1.1, rc={"figure.figsize": (8, 3)})
sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})

# ===========================================================================================
# PART 1: Setting up the SceneDataLoader
# ===========================================================================================

"""
The dataset is structured as a collection of traffic scenarios.
Each file beginning with 'tfrecord' is a unique traffic scenario.

SceneDataLoader helps us load and iterate through these scenarios
following PyTorch dataloader conventions.
"""

from gpudrive.env.dataset import SceneDataLoader

# Initialize the data loader
data_loader = SceneDataLoader(
    root="data/processed/examples",  # Path to the dataset
    batch_size=10,  # Batch size, should equal the number of worlds (envs)
    dataset_size=4,  # Total number of different scenes to use
    sample_with_replacement=True,  # Whether to sample with replacement
    seed=42,  # Random seed for reproducibility
    shuffle=True,  # Whether to shuffle the dataset
)

# Display the full dataset we'll be using
print("Full dataset:", data_loader.dataset)

# Display unique scenes (should be 4 as specified by dataset_size)
print("Unique scenes:", set(data_loader.dataset))

# Get the first batch of data files
data_files = next(iter(data_loader))
print("First data file:", data_files[0])

# ===========================================================================================
# PART 2: Setting up the GPUDrive Environment
# ===========================================================================================

from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.config import EnvConfig

# Pass the data_loader to the environment
env = GPUDriveTorchEnv(
    config=EnvConfig(),
    data_loader=data_loader,
    max_cont_agents=64,  # Maximum number of continuous agents
    device="cpu",  # Using CPU for computation
)

# ===========================================================================================
# PART 3: Deep Dive into Traffic Scenario Structure
# ===========================================================================================

"""
Traffic scenarios are essentially dictionaries containing key elements:
- Road map: Layout and structure of roads
- Human driving (expert) demonstrations
- Road objects: Elements such as stop signs and traffic signals
"""

# Take an example scene to explore
data_path = "data/processed/examples/tfrecord-00000-of-01000_222.json"

with open(data_path) as file:
    traffic_scene = json.load(file)

# Display the main keys in the traffic scene
print("\nTraffic scene main components:", traffic_scene.keys())

# ===========================================================================================
# PART 4: Understanding the Scene Components
# ===========================================================================================

"""
Global Overview of a traffic scene:
- name: Name of the traffic scenario
- scenario_id: Unique identifier for every scenario
- objects: Dynamic entities such as vehicles or other moving elements
- roads: Stationary elements, including road points and fixed objects
- tl_states: Traffic light states (currently not included in processing)
- metadata: Additional details about the scenario
"""

print("\nScene name:", traffic_scene["name"])
print("Scenario ID:", traffic_scene["scenario_id"])
print("Traffic light states:", traffic_scene["tl_states"])
print("Metadata:", traffic_scene["metadata"])

# ===========================================================================================
# PART 5: Visualizing Road Objects
# ===========================================================================================

# Create a chart showing the distribution of road objects
def visualize_objects():
    """Visualize the distribution of objects in the traffic scene."""
    pd.Series(
        [
            traffic_scene["objects"][idx]["type"]
            for idx in range(len(traffic_scene["objects"]))
        ]
    ).value_counts().plot(kind="bar", rot=45, color=cmap)
    plt.title(
        f'Distribution of road objects in traffic scene. Total # objects: {len(traffic_scene["objects"])}'
    )
    plt.tight_layout()
    plt.savefig("object_distribution.png")
    plt.close()

# Create a chart showing the distribution of road points
def visualize_road_points():
    """Visualize the distribution of road points in the traffic scene."""
    pd.Series(
        [traffic_scene["roads"][idx]["type"] for idx in range(len(traffic_scene["roads"]))]
    ).value_counts().plot(kind="bar", rot=45, color=cmap)
    plt.title(
        f'Distribution of road points in traffic scene. Total # points: {len(traffic_scene["roads"])}'
    )
    plt.tight_layout()
    plt.savefig("road_points_distribution.png")
    plt.close()

# Run visualizations
visualize_objects()
visualize_road_points()

# ===========================================================================================
# PART 6: Exploring Road Objects in Detail
# ===========================================================================================

"""
Road objects contain details about:
- Position: (x, y, z) coordinates at every time step
- Width and length: Size of the object
- Heading: Direction the object is pointing
- Velocity: Speed in x and y directions
- Valid: Whether the state was observed at each time point
- Goal position: Final position of the vehicle
- Type: Vehicle, cyclist, etc.
"""

# Examine the first object in detail
idx = 0
object_example = traffic_scene["objects"][idx]

print("\nObject Keys:", object_example.keys())
print("\nObject Type:", object_example["type"])
print("Object Dimensions (width, length):", object_example["width"], object_example["length"])
print("Object Goal Position:", object_example["goalPosition"])

# Show first 5 position entries to demonstrate structure
print("\nFirst 5 position entries:")
for i, pos in enumerate(object_example["position"][:5]):
    print(f"Timestep {i}: x={pos['x']:.2f}, y={pos['y']:.2f}, z={pos['z']:.2f}")

# ===========================================================================================
# PART 7: Visualizing Object Heading and Validity
# ===========================================================================================

def visualize_heading():
    """Visualize the heading of an object over time."""
    plt.figure(figsize=(8, 3))
    plt.plot(object_example["heading"])
    plt.xlabel("Time step")
    plt.ylabel("Heading")
    plt.title("Vehicle Heading Over Time")
    plt.tight_layout()
    plt.savefig("heading_over_time.png")
    plt.close()

def visualize_validity():
    """Visualize whether the object state was observed at each time point."""
    plt.figure(figsize=(8, 3))
    plt.plot(object_example["valid"], "_", lw=5)
    plt.xlabel("Time step")
    plt.ylabel("IS VALID")
    plt.title("Object State Validity Over Time")
    plt.tight_layout()
    plt.savefig("validity_over_time.png")
    plt.close()

# Run heading and validity visualizations
visualize_heading()
visualize_validity()

# ===========================================================================================
# PART 8: Exploring Road Points in Detail
# ===========================================================================================

"""
Road points represent static objects in the scene:
- Geometry: (x, y, z) position(s) for a road point
- Type: lane, road_edge, stop_sign, etc.
- Map element ID: Identifier for the map element
- ID: Unique identifier for the road point
"""

road_idx = 0
road_example = traffic_scene["roads"][road_idx]

print("\nRoad Point Keys:", road_example.keys())
print("Road Point Type:", road_example["type"])

# Show first 3 geometry entries to demonstrate structure
print("\nFirst 3 geometry entries for the road point:")
for i, geo in enumerate(road_example["geometry"][:3]):
    print(f"Point {i}: x={geo['x']:.2f}, y={geo['y']:.2f}, z={geo['z']:.2f}")

# ===========================================================================================
# PART 9: Main Function
# ===========================================================================================

def main():
    """Main function to run the script."""
    print("\nGPUDrive Scenario Analysis Complete")
    print("Visualization files created: ")
    print("- object_distribution.png")
    print("- road_points_distribution.png")
    print("- heading_over_time.png")
    print("- validity_over_time.png")

if __name__ == "__main__":
    main()