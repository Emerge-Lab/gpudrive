import re
import os
import yaml
from scripts.sim_utils.create import SimCreator
import build


def get_constants():
    fullpath = os.path.join(os.path.dirname(__file__), "config.yml")
    sim = SimCreator(fullpath)
    consts = sim.shape_tensor().to_torch()[0].flatten().tolist()
    print(consts)
    return consts

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