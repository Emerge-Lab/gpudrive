import numpy as np

"""Predefined constants for the environment."""

# Observation space constants
MAX_SPEED = 100
MAX_VEH_LEN = 30
MAX_VEH_WIDTH = 15
MAX_VEH_HEIGHT = 5
MIN_REL_GOAL_COORD = (
    -10_000
)  # Note: (1000 or 100 should be enough, may indicate a bug or these are just from dead controlled agents)
MAX_REL_GOAL_COORD = 10_000
MIN_REL_AGENT_POS = -1000
MAX_REL_AGENT_POS = 1000
MAX_ORIENTATION_RAD = 2 * np.pi

# Road graph constants
MIN_RG_COORD = -1000
MAX_RG_COORD = 1000
MAX_ROAD_LINE_SEGMENT_LEN = 100
MAX_ROAD_SCALE = 100

# Feature shape constants
EGO_FEAT_DIM = 6
PARTNER_FEAT_DIM = 6
ROAD_GRAPH_FEAT_DIM = 13
