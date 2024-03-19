from gpudrive import SimManager
from sim_utils.create import SimCreator

import json
import os
import shutil
import yaml
import csv
from tqdm import tqdm
import pandas as pd

VALID_FILES_PATH = "/home/aarav/nocturne_data/formatted_json_v2_no_tl_valid"
BINNING_TEMP_PATH = "/home/aarav/nocturne_data/binning_temp"
FINAL_BINNED_JSON_PATHS = "/home/aarav/nocturne_data/binned_jsons"
CSV_PATH = "/home/aarav/nocturne_data/datacsv.csv"

def modify_valid_files_json(valid_files_path: str, file_path: str):
    if(os.path.exists(valid_files_path + "/valid_files.json") == False):
        with open(valid_files_path + "/valid_files.json", 'w') as file:
            json.dump({}, file)
    with open(valid_files_path + "/valid_files.json", 'r') as file:
        valid_files = json.load(file)
    valid_files.clear()
    valid_files[file_path] = []
    with open(valid_files_path + "/valid_files.json", 'w') as file:
        json.dump(valid_files, file)

def delete_file_from_dest(file_path: str):
    os.remove(file_path)

def copy_file_to_dest(file_path: str, dest_path: str):
    shutil.copy(file_path, dest_path)
    return os.path.join(dest_path, os.path.basename(file_path))

def return_list_of_files(valid_files_path: str):
    with open(valid_files_path + "/valid_files.json", 'r') as file:
        valid_files = json.load(file)
    file_list = []
    for file in valid_files:
        file_list.append(os.path.join(valid_files_path, file))
    return file_list

# def return_agent_numbers(sim: SimManager):
#     shape = sim.shape_tensor().to_torch()
#     num_agents, num_roads = shape[0].flatten().tolist()
#     return num_agents, num_roads

def return_agent_numbers(file_path: str):
    with open(file_path, 'r') as file:
        data = json.load(file)
    num_agents = len(data['objects'])
    num_roads = len(data['roads'])
    num_road_segments = 0
    for road in data['roads']:
        if(road['type'] == "road_edge" or road['type'] == "road_line" or road['type'] == "lane"):
            num_road_segments += len(road['geometry']) - 1
        else:
            num_road_segments += 1
    return num_agents, num_road_segments

if __name__ == "__main__":
    # with open("config.yml", 'r') as file:
    #     config = yaml.safe_load(file)
    # config['sim_manager']['num_worlds'] = 1
    # config['sim_manager']['exec_mode'] = "CPU"
    # config['sim_manager']['json_path'] = BINNING_TEMP_PATH
    # file_list = return_list_of_files(VALID_FILES_PATH)
    # file_meta_data = []
    # file_meta_data.append(["File Path", "Number of Agents", "Number of Roads"])
    # for file in tqdm(file_list):
    #     # currfile = copy_file_to_dest(file, BINNING_TEMP_PATH)
    #     # modify_valid_files_json(BINNING_TEMP_PATH, file)
    #     num_entities = return_agent_numbers(file)
    #     file_meta_data.append([file, num_entities[0], num_entities[1]])
    #     # delete_file_from_dest(currfile)

    # with open(CSV_PATH, 'w') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(file_meta_data)

    data = pd.read_csv(CSV_PATH)
    sorted_data = data.sort_values('Number of Agents')

    bins = []
    bin_size = 100
    number_of_bins = len(sorted_data) // bin_size + (1 if len(sorted_data) % bin_size > 0 else 0)

    # Create bins of 100 files each
    for i in range(number_of_bins):
        bin_start = i * bin_size
        bin_end = min((i + 1) * bin_size, len(sorted_data))
        bins.append(sorted_data.iloc[bin_start:bin_end])

    if not os.path.exists(FINAL_BINNED_JSON_PATHS):
        os.makedirs(FINAL_BINNED_JSON_PATHS)
    
    for i, bin in enumerate(bins):
        if not os.path.exists(FINAL_BINNED_JSON_PATHS + f"/bin_{i}"):
            os.makedirs(FINAL_BINNED_JSON_PATHS + f"/bin_{i}")
        bin_folder = FINAL_BINNED_JSON_PATHS + f"/bin_{i}"
        print(bin_folder)
        d = {}
        for index, row in bin.iterrows():
            file_path = row['File Path']
            d[file_path] = [row['Number of Agents'], row['Number of Roads']]
        filepath = os.path.join(bin_folder, f"valid_files.json")
        print(filepath)
        with open(filepath, 'w') as file:
            json.dump(d, file)
    print("Binning complete")