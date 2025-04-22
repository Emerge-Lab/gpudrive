import numpy as np

"""Predefined constants for the environment."""

# Observation space constants
MAX_SPEED = 100
MAX_VEH_LEN = 30
MAX_VEH_WIDTH = 15
MAX_VEH_HEIGHT = 10
MIN_REL_GOAL_COORD = -1000
MAX_REL_GOAL_COORD = 1000
MIN_REL_AGENT_POS = -1000
MAX_REL_AGENT_POS = 1000
MAX_ORIENTATION_RAD = 2 * np.pi

MAX_REF_POINT = 1000

# Road graph constants
MIN_RG_COORD = -1000
MAX_RG_COORD = 1000
MAX_ROAD_LINE_SEGMENT_LEN = 100
MAX_ROAD_SCALE = 100

# Feature shape constants
EGO_FEAT_DIM = 6  # Ego state base fields
PARTNER_FEAT_DIM = 6
ROAD_GRAPH_FEAT_DIM = 13

# Dataset constants
LOG_TRAJECTORY_LEN = 91

# BEV observation constants
BEV_RASTERIZATION_RESOLUTION = 200
NUM_MADRONA_ENTITY_TYPES = 11

# Action values
MAX_ACTION_VALUE = 4.0

# Invalid points
INVALID_ID = -1.0