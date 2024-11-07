import waymax
import numpy as np
import math
import mediapy
from tqdm import tqdm
import dataclasses
import os
from pathlib import Path
import pickle
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from waymax import config as _config
from waymax import dataloader, datatypes, visualization, dynamics
from waymax.datatypes.simulator_state import SimulatorState
from waymax.config import EnvironmentConfig, ObjectType

# Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != "gpudrive":
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)

# VBD dependencies
from integrations.models.vbd.sim_agent.waymax_env import WaymaxEnvironment
from integrations.models.vbd.data.dataset import WaymaxTestDataset
from integrations.models.vbd.waymax_visualization.plotting import plot_state
from integrations.models.vbd.sim_agent.sim_actor import (
    VBDTest,
    sample_to_action,
)
from integrations.models.vbd.model.utils import set_seed


DATA_DIR = "data/processed/waymax"  # Base data path
CKPT_DIR = "data/checkpoints"  # Base checkpoint path

SCENARIO_ID = "11671609ebfa3185"  # Debugging scenario we use
CKPT_PATH = "vbd/weights/epoch=18.ckpt"

FPS = 20
INIT_STEPS = 11  # Warmup period
MAX_CONTROLLED_OBJECTS = 32

# Load model
model = VBDTest.load_from_checkpoint(CKPT_PATH, torch.device("cpu"))
_ = model.cuda()
_ = model.eval()

# Model settings
replan_freq = 80  # Roll out every X steps 80 means openloop
model.early_stop = 0  # Stop Diffusion Early From 100 to X
model.skip = 1  # Skip Alpha
model.reward_func = None

env_config = EnvironmentConfig(
    controlled_object=ObjectType.VALID,
    allow_new_objects_after_warmup=False,
    init_steps=INIT_STEPS + 1,
    max_num_objects=MAX_CONTROLLED_OBJECTS,
)

waymax_env = WaymaxEnvironment(
    dynamics_model=dynamics.StateDynamics(),
    config=env_config,
    log_replay=True,
)

scenario_path = os.path.join(DATA_DIR, SCENARIO_ID + ".pkl")
with open(f"{DATA_DIR}/waymax_scenario_{SCENARIO_ID}.pkl", "rb") as f:
    scenario = pickle.load(f)

# Create "dataset" (need for utils)
dataset = WaymaxTestDataset(
    data_dir="data/processed/waymax",
    anchor_path="vbd/data/cluster_64_center_dict.pkl",
    max_object=MAX_CONTROLLED_OBJECTS,
)


if __name__ == "__main__":

    # Reset
    init_state = waymax_env.reset(scenario)

    # Process the scenario
    current_state = init_state
    sample = dataset.process_scenario(
        init_state, current_index=init_state.timestep, use_log=False
    )
