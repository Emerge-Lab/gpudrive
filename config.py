import re
import os

from scripts.sim_utils.create import SimCreator
import build

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
    print("Hello World!")

if __name__ == "__main__":
    update_constants("src/consts.hpp", 200, 2000)