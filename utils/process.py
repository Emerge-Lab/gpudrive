import os
import yaml
import json
import numpy as np
from numpy.random import default_rng




def generate_valid_files_json(
    num_unique_scenes, data_dir, output_file_name="valid_files.json"
):
    """
    Create a valid_files.json file for the given number of unique scenes.
    """

    # Dictionary to store the filenames
    file_dict = {}

    # Loop through all files in the folder
    for filename in os.listdir(data_dir):
        if filename.startswith("tfrecord") and filename.endswith(".json"):
            file_dict[filename] = []

    # Sample n random files
    rng = default_rng()
    rand_int = rng.choice(a=len(file_dict), size=num_unique_scenes, replace=False)

    # create a new dictionary with only the items from the random indices
    file_dict = {
        list(file_dict.keys())[i]: file_dict[list(file_dict.keys())[i]]
        for i in rand_int
    }

    # Write the dictionary to a YAML file
    # This will overwrite the current json file if it exists
    with open(os.path.join(data_dir, output_file_name), "w") as json_file:
        json.dump(
            file_dict,
            json_file,
            indent=4,
        )

    print(
        f"Generated valid_files.json with {num_unique_scenes} unique scene(s): \n {file_dict}"
    )
    
    return len(file_dict)
    
    
