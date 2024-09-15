import sys
import os
import torch
import pickle
import glob
import numpy as np
from torch.utils.data import Dataset
from .data_utils import *
import functools
import pickle


class WaymaxDataset(Dataset):
    """
    Dataset class for Waymax data.

    Args:
        data_dir (str): Directory path where the data is stored.
        anchor_path (str, optional): Path to the anchor file. Defaults to "data/cluster_64_center_dict.pkl".
    """

    def __init__(
        self,
        data_dir,
        anchor_path="data/cluster_64_center_dict.pkl",
    ):
        self.data_list = glob.glob(data_dir + "/*")
        print(f"Found {len(self.data_list)} scenarios")
        self.anchors = pickle.load(open(anchor_path, "rb"))

        self.__collate_fn__ = data_collate_fn

    def __len__(self):
        return len(self.data_list)

    def _process(self, types):
        """
        Process the agent types and convert them into anchor vectors.

        Args:
            types (numpy.ndarray): Array of agent types.

        Returns:
            numpy.ndarray: Array of anchor vectors.
        """
        anchors = []

        for i in range(len(types)):
            if types[i] == 1:
                anchors.append(self.anchors["TYPE_VEHICLE"])
            elif types[i] == 2:
                anchors.append(self.anchors["TYPE_PEDESTRIAN"])
            elif types[i] == 3:
                anchors.append(self.anchors["TYPE_CYCLIST"])
            else:
                anchors.append(np.zeros_like(self.anchors["TYPE_VEHICLE"]))

        return np.array(anchors, dtype=np.float32)

    def gen_tensor(self, data):
        """
        Generate tensors from the input data.

        Args:
            data (dict): Input data dictionary.

        Returns:
            dict: Dictionary of tensors.
        """

        agents_history = data["agents_history"]
        agents_interested = data["agents_interested"]
        agents_future = data["agents_future"]
        agents_type = data["agents_type"]
        traffic_light_points = data["traffic_light_points"]
        polylines = data["polylines"]
        polylines_valid = data["polylines_valid"]
        relations = data["relations"]
        anchors = self._process(agents_type)

        tensors = {
            "agents_history": torch.from_numpy(agents_history),
            "agents_interested": torch.from_numpy(agents_interested),
            "agents_future": torch.from_numpy(agents_future),
            "agents_type": torch.from_numpy(agents_type),
            "traffic_light_points": torch.from_numpy(traffic_light_points),
            "polylines": torch.from_numpy(polylines),
            "polylines_valid": torch.from_numpy(polylines_valid),
            "relations": torch.from_numpy(relations),
            "anchors": torch.from_numpy(anchors),
        }

        return tensors

    def __getitem__(self, idx):
        with open(self.data_list[idx], "rb") as f:
            data = pickle.load(f)
        return self.gen_tensor(data)


class WaymaxTestDataset(WaymaxDataset):
    """
    Test dataset class for Waymax data.

    Args:
        data_dir (str): Directory path where the data is stored.
        anchor_path (str, optional): Path to the anchor file. Defaults to "data/cluster_64_center_dict.pkl".
        max_object (int, optional): Maximum number of objects. Defaults to 16.
        max_polylines (int, optional): Maximum number of polylines. Defaults to 256.
        history_length (int, optional): Length of history. Defaults to 11.
        num_points_polyline (int, optional): Number of points in each polyline. Defaults to 30.
    """

    def __init__(
        self,
        data_dir: str,
        anchor_path="data/cluster_64_center_dict.pkl",
        max_object: int = 16,
        max_map_points: int = 3000,
        max_polylines: int = 256,
        history_length: int = 11,
        num_points_polyline: int = 30,
    ) -> None:
        super().__init__(data_dir, anchor_path)

        self.max_object = max_object
        self.max_polylines = max_polylines
        self.max_map_points = max_map_points
        self.history_length = history_length
        self.num_points_polyline = num_points_polyline

        self.base_path = os.path.dirname(os.path.abspath(self.data_list[0]))

    def process_scenario(
        self,
        scenario_raw,
        current_index: int = 10,
        use_log: bool = True,
        selected_agents=None,
        remove_history=False,
    ):
        """
        Process a scenario and generate tensors.

        Args:
            scenario_raw (dict): Raw scenario data.
            current_index (int, optional): Current index. Defaults to 10.
            use_log (bool, optional): Whether to use logarithmic scaling. Defaults to True.
            selected_agents (list, optional): List of selected agents. Defaults to None.

        Returns:
            dict: Dictionary of tensors.
        """
        data_dict = data_process_scenario(
            scenario_raw,
            current_index=current_index,
            max_num_objects=self.max_object,
            max_polylines=self.max_polylines,
            num_points_polyline=self.num_points_polyline,
            use_log=use_log,
            selected_agents=selected_agents,
            remove_history=remove_history,
        )

        data_dict["anchors"] = self._process(data_dict["agents_type"])

        return data_dict

    def reset_agent_length(self, max_object):
        """
        Reset the maximum number of objects.

        Args:
            max_object (int): Maximum number of objects.
        """
        self.max_object = max_object

    def get_scenario_by_id(
        self,
        scenario_id,
        current_index: int = 10,
        use_log: bool = True,
        remove_history=False,
    ):
        """
        Get a scenario by its ID.

        Args:
            scenario_id (int): Scenario ID.
            current_index (int, optional): Current index. Defaults to 10.
            use_log (bool, optional): Whether to use logarithmic scaling. Defaults to True.

        Returns:
            tuple: Scenario ID, scenario raw data, and tensors.
        """
        file_path = os.path.join(self.base_path, f"scenario_{scenario_id}.pkl")
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        try:
            scenario_raw = data["scenario_raw"]
        except:
            print(data.keys())
            raise ValueError("scenario_raw not found")

        data_dict = self.process_scenario(
            scenario_raw,
            current_index=current_index,
            use_log=use_log,
            remove_history=remove_history,
        )

        return scenario_id, scenario_raw, data_dict

    def get_scenario_by_index(
        self,
        index,
        current_index: int = 10,
        use_log: bool = True,
        remove_history=False,
    ):
        """
        Get a scenario by its index.

        Args:
            index (int): Scenario index.
            current_index (int, optional): Current index. Defaults to 10.
            use_log (bool, optional): Whether to use logarithmic scaling. Defaults to True.

        Returns:
            tuple: Scenario ID, scenario raw data, and tensors.
        """
        with open(self.data_list[index], "rb") as f:
            data = pickle.load(f)

        scenario_raw = data["scenario_raw"]
        scenario_id = data["scenario_id"]

        data_dict = self.process_scenario(
            scenario_raw,
            current_index=current_index,
            use_log=use_log,
            remove_history=remove_history,
        )

        return scenario_id, scenario_raw, data_dict

    def __getitem__(self, idx):
        _, _, data_dict = self.get_scenario_by_index(idx)
        return data_dict
