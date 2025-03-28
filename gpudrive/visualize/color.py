import madrona_gpudrive
import numpy as np

dark_red = np.array([139, 0, 0]) / 255.0
dark_grey = "#000000"  # np.array([80, 80, 80]) / 255.0

ROAD_GRAPH_COLORS = {
    1: dark_grey,  # 'RoadEdgeBoundary' (Gray)
    2: np.array([120, 120, 120])
    / 255.0,  # 'RoadLine-BrokenSingleYellow' (Light Purple)
    3: np.array([230, 230, 230]) / 255.0,  # 'LaneCenter-Freeway' (Light Gray)
    4: np.array([200, 200, 200]) / 255.0,  # 'Crosswalk' (Light Gray)
    5: np.array([0.85, 0.65, 0.13]),  # 'SpeedBump' (Dark yellow)
    6: np.array([255, 0, 0]) / 255.0,  # 'StopSign' (Red)
}

ROAD_GRAPH_TYPE_NAMES = {  # 1-6 ; 0 is None (padding)
    int(madrona_gpudrive.EntityType.RoadEdge): "Road edge",
    int(madrona_gpudrive.EntityType.RoadLine): "Road line",
    int(madrona_gpudrive.EntityType.RoadLane): "Lane center",
    int(madrona_gpudrive.EntityType.CrossWalk): "Crosswalk",
    int(madrona_gpudrive.EntityType.SpeedBump): "Speed bump",
    int(madrona_gpudrive.EntityType.StopSign): "Stop sign",
}

AGENT_COLOR_BY_STATE = {
    "ok": "#4B77BE",  # Controlled and doing fine
    "collided": "r",  # Controlled and collided
    "off_road": "orange",  # Controlled and off-road
    "log_replay": "#c7c7c7",  # Agents marked as expert controlled or static
}

REL_OBS_OBJ_COLORS = {
    "ego": "#0066ff",
    "ego_goal": "#0099cc",
    "other_agents": "#ff884d",
}
AGENT_COLOR_BY_POLICY= ['g','b','p']