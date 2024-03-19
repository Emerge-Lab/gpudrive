import re
import os
import yaml
from scripts.sim_utils.create import SimCreator
import build
import torch

def get_constants():
    fullpath = os.path.join(os.path.dirname(__file__), "config.yml")
    with open(fullpath, 'r') as file:
        config = yaml.safe_load(file)
    sim = SimCreator(config)
    consts = sim.shape_tensor().to_torch()
    max_agents = torch.max(consts[:, 0], 0).values.item()
    max_roads = torch.max(consts[:, 1], 0).values.item()
    print(consts)
    print(max_agents, max_roads)
    return max_agents, max_roads

def update_constants(filepath, new_max_agent_count, new_max_road_entity_count):
    with open(filepath, 'r') as file:
        content = file.read()

    content = re.sub(
        r"(inline constexpr madrona::CountT kMaxAgentCount = )\d+;",
        r"\g<1>{};".format(new_max_agent_count),
        content)

    content = re.sub(
        r"(inline constexpr madrona::CountT kMaxRoadEntityCount = )\d+;",
        r"\g<1>{};".format(new_max_road_entity_count),
        content)

    # Write the updated content back to the file
    with open(filepath, 'w') as file:
        file.write(content)

    build.build()

def main():
    new_max_agent_count, new_max_road_entity_count = get_constants()
    update_constants("src/consts.hpp", new_max_agent_count, new_max_road_entity_count+1)

if __name__ == "__main__":
    update_constants("src/consts.hpp", 200, 2000)