import gpudrive
import numpy as np

dark_red = np.array([139, 0, 0]) / 255.0
dark_grey = np.array([80, 80, 80]) / 255.0
black = np.array([0,0,0])/ 255.0
blue_green = np.array([117,181,197])/255.0

ROAD_GRAPH_COLORS = {
    1: black,  # 'RoadEdgeBoundary' (Gray)
    2: np.array([120, 120, 120])
    / 255.0,  # 'RoadLine-BrokenSingleYellow' (Light Purple)
    3: np.array([230, 230, 230]) / 255.0,  # 'LaneCenter-Freeway' (Light Gray)
    4: np.array([200, 200, 200]) / 255.0,  # 'Crosswalk' (Light Gray)
    5: np.array([0.85, 0.65, 0.13]),  # 'SpeedBump' (Dark yellow)
    6: np.array([255, 0, 0]) / 255.0  # 'StopSign' (Red)
    # 7: np.array([117,181,197])/ 255.0, # "controlled agents" same as the one in waymax controlled agent
    # 8: np.array([117,181,197])/ 255.0
}

ROAD_GRAPH_TYPE_NAMES = {  # 1-6 ; 0 is None (padding)
    int(gpudrive.EntityType.RoadEdge): "Road edge",
    int(gpudrive.EntityType.RoadLine): "Road line",
    int(gpudrive.EntityType.RoadLane): "Lane center",
    int(gpudrive.EntityType.CrossWalk): "Crosswalk",
    int(gpudrive.EntityType.SpeedBump): "Speed bump",
    int(gpudrive.EntityType.StopSign): "Stop sign",
}

REL_OBS_OBJ_COLORS = {
    "ego": "#4169E1",
    "ego_goal": "#91a8ee",
    "other_agents": "#e60000",
}
