import numpy as np

ROAD_GRAPH_COLORS = {
    1: np.array([80, 80, 80]) / 255.0,  # 'RoadEdgeBoundary' (Gray)
    2: np.array([120, 120, 120])
    / 255.0,  # 'RoadLine-BrokenSingleYellow' (Light Purple)
    3: np.array([230, 230, 230]) / 255.0,  # 'LaneCenter-Freeway' (Light Gray)
    4: np.array([200, 200, 200]) / 255.0,  # 'Crosswalk' (Light Gray)
    5: np.array([0.85, 0.65, 0.13]),  # 'SpeedBump' (Dark yellow)
    6: np.array([255, 0, 0]) / 255.0,  # 'StopSign' (Red)
}

ROAD_GRAPH_TYPE_NAMES = {
    1: "Road edge",
    2: "Road line",
    3: "Lane center",
    4: "Crosswalk",
    5: "Speed bump",
    6: "Stop sign",
}

REL_OBS_OBJ_COLORS = {
    "ego": "#4169E1",
    "ego_goal": "#91a8ee",
    "other_agents": "#e60000",
}
