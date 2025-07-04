import os
from pathlib import Path
import mediapy

# Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != 'GPUDrive-Fork':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)

from gpudrive.env.config import EnvConfig, SceneConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
import sys
from tqdm import tqdm
import time

DYNAMICS_MODEL = "delta_local" # "delta_local" / "state" / "classic"
DATA_PATH =  sys.argv[1]# Your data path
MAX_NUM_OBJECTS = 64
NUM_ENVS = 1
NUM_SCENES = 1000

# Configs
env_config = EnvConfig(dynamics_model=DYNAMICS_MODEL)

# # Make dataloader
data_loader = SceneDataLoader(
    root=DATA_PATH, # Path to the dataset
    batch_size=NUM_ENVS, # Batch size, you want this to be equal to the number of worlds (envs) so that every world receives a different scene
    dataset_size=NUM_SCENES, # Total number of different scenes we want to use
    sample_with_replacement=False, 
    seed=42, 
    shuffle=True,
    file_prefix=sys.argv[2]
)

# Make environment
env = GPUDriveTorchEnv(
    config=env_config,
    data_loader=data_loader,
    max_cont_agents=MAX_NUM_OBJECTS, # Maximum number of agents to control per scenario
    device="cuda", 
    action_type="continuous" # "continuous" or "discrete"
)

done_envs = []
render = False
frames = {f"env_{i}": [] for i in range(NUM_ENVS)}
times = []
batch_swap_times = []
# Step through the scene
for i in range(0, data_loader.dataset_size, NUM_ENVS):

    if i < data_loader.dataset_size - NUM_ENVS:
        print("Swapping batch: ", i)
        batch_swap_start = time.time()
        env.swap_data_batch()
        batch_swap_elapsed = time.time() - batch_swap_start
        batch_swap_times.append(batch_swap_elapsed)

print(f"Ran {NUM_SCENES} scenes")
print(f"Batch swap total time: {sum(batch_swap_times)}s")
print(f"Batch swap average time: {sum(batch_swap_times) / len(batch_swap_times)}s")