from huggingface_hub import snapshot_download
import gpudrive
import random
import os
import json
from tqdm import tqdm  # Import tqdm for progress bar


def is_valid_json_structure(file_path):
    """Check if the JSON file has the expected structure."""
    try:
        with open(file_path, "r") as file:
            data = json.load(file)

        # Check if required keys exist in the top-level structure
        if not all(
            key in data for key in ["name", "objects", "roads", "tl_states"]
        ):
            return False

        # Check if "objects" is a list and contains dictionaries with the correct structure
        if not isinstance(data["objects"], list) or not all(
            isinstance(obj, dict) and "position" in obj and "type" in obj
            for obj in data["objects"]
        ):
            return False

        # Check if "roads" is a list of dictionaries with required "geometry"
        if not isinstance(data["roads"], list) or not all(
            isinstance(road, dict) and "geometry" in road
            for road in data["roads"]
        ):
            return False

        # Optionally, check that each "geometry" in "roads" has valid "x" and "y" coordinates
        for road in data["roads"]:
            if not all(
                isinstance(geo, dict) and "x" in geo and "y" in geo
                for geo in road.get("geometry", [])
            ):
                return False

        return True

    except (json.JSONDecodeError, ValueError, IOError) as e:
        # If an error occurs (invalid JSON or IO error), mark as invalid
        return False


def delete_corrupted_file(file_path):
    """Deletes the corrupted file."""
    try:
        os.remove(file_path)
        print(f"Deleted corrupted file: {file_path}")
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")


if __name__ == "__main__":

    data_dir = "data/processed/training"

    total_scenes = 0
    corrupted_scenes = 0

    # Get the list of all json files in the directory
    json_files = [
        scene for scene in os.listdir(data_dir) if scene.endswith(".json")
    ]

    # Initialize tqdm progress bar
    with tqdm(
        total=len(json_files), desc="Processing scenes", colour="green"
    ) as pbar:

        # Iterate through the files in the directory
        for scene in json_files:
            scene_path = os.path.abspath(os.path.join(data_dir, scene))

            total_scenes += 1

            # Check if the file is corrupted
            if not is_valid_json_structure(scene_path):
                print(f"Corrupted file detected: {scene_path}")
                delete_corrupted_file(scene_path)
                corrupted_scenes += 1

            pbar.update(
                1
            )  # Update progress bar by 1 after processing each file

    print(f"Total number of scenes: {total_scenes}")
    print(f"Total number of corrupted scenes: {corrupted_scenes}")
