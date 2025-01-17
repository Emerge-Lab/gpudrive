canvas_config = {
    "width": 10,
    "background_color": "xkcd:white",
    "tick_on": True,
}

road_line_config = {
    "TYPE_UNKNOWN": {
        "color": "xkcd:light grey",
        "linewidth": 3,
        "linestyle": "dotted",
        "alpha": 1,
    },
    "TYPE_BROKEN_SINGLE_WHITE": {
        "color": "xkcd:medium grey",
        "linewidth": 2,
        "linestyle": "-",
        "alpha": 1,
    },
    "TYPE_SOLID_SINGLE_WHITE": {
        "color": "xkcd:medium grey",
        "linewidth": 2,
        "linestyle": "-",
        "alpha": 1,
    },
    "TYPE_SOLID_DOUBLE_WHITE": {
        "color": "xkcd:medium grey",
        "linewidth": 3.5,
        "linestyle": "-",
        "alpha": 1,
    },
    "TYPE_BROKEN_SINGLE_YELLOW": {
        "color": "xkcd:yellowish orange",
        "linewidth": 2,
        "linestyle": "-",
        "alpha": 1,
    },
    "TYPE_BROKEN_DOUBLE_YELLOW": {
        "color": "xkcd:yellowish orange",
        "linewidth": 3.5,
        "linestyle": "-",
        "alpha": 1,
    },
    "TYPE_SOLID_SINGLE_YELLOW": {
        "color": "xkcd:yellowish orange",
        "linewidth": 4,
        "linestyle": "-",
        "alpha": 1,
    },
    "TYPE_SOLID_DOUBLE_YELLOW": {
        "color": "xkcd:yellowish orange",
        "linewidth": 5,
        "linestyle": "-",
        "alpha": 1,
    },
    "TYPE_PASSING_DOUBLE_YELLOW": {
        "color": "xkcd:yellowish orange",
        "linewidth": 3,
        "linestyle": "-.",
        "alpha": 1,
    },
}

road_edge_config = {
    "TYPE_UNKNOWN": {
        "color": "xkcd:brown",
        "linewidth": 3,
        "linestyle": "dotted",
        "alpha": 1,
    },
    "TYPE_ROAD_EDGE_BOUNDARY": {
        "color": "xkcd:charcoal",
        "linewidth": 2.5,
        "linestyle": "-",
        "alpha": 0.8,
    },
    "TYPE_ROAD_EDGE_MEDIAN": {
        "color": "xkcd:sage",
        "linewidth": 3.5,
        "linestyle": "-",
        "alpha": 1,
    },
}

lane_config = {
    "TYPE_UNDEFINED": {
        "color": "xkcd:light grey",
        "linewidth": 3,
        "linestyle": "dotted",
        "alpha": 1,
    },
    "TYPE_FREEWAY": {
        "color": "xkcd:light blue",
        "linewidth": 3,
        "linestyle": "dotted",
        "alpha": 1,
    },
    "TYPE_SURFACE_STREET": {
        "color": "xkcd:light khaki",
        "linewidth": 3,
        "linestyle": "dotted",
        "alpha": 1,
    },
    "TYPE_BIKE_LANE": {
        "color": "xkcd:light mint",
        "linewidth": 3,
        "linestyle": "dotted",
        "alpha": 1,
    },
}

speed_bump_config = {
    "facecolor": "xkcd:goldenrod",
    "edgecolor": "xkcd:black",
    "alpha": 1,
}

crosswalk_config = {
    "facecolor": "None",
    "edgecolor": "xkcd:bluish grey",
    "alpha": 0.4,
}

stop_sign_config = {
    "facecolor": "xkcd:red",
    "edgecolor": "none",
    "linewidth": 3,
    "radius": 3,
    "alpha": 1,
}

object_config = {
    "TYPE_UNSET": {
        "facecolor": "xkcd:black",
        "linewidth": 2,
        "linestyle": "-",
        "alpha": 0.5,
    },
    "TYPE_VEHICLE": {
        "facecolor": "xkcd:blue",
        "linewidth": 4,
        "linestyle": "-",
        "alpha": 0.5,
    },
    "TYPE_PEDESTRIAN": {
        "facecolor": "xkcd:green",
        "linewidth": 2,
        "linestyle": "-",
        "alpha": 0.5,
    },
    "TYPE_CYCLIST": {
        "facecolor": "xkcd:yellow",
        "linewidth": 3,
        "linestyle": "-",
        "alpha": 0.5,
    },
    "TYPE_OTHER": {
        "facecolor": "xkcd:black",
        "linewidth": 3,
        "linestyle": "-",
        "alpha": 0.5,
    },
}

signal_config = {
    "linewidth": 2,
    "radius": 1.5,
    "LANE_STATE_UNKNOWN": {
        "facecolor": "xkcd:light grey",
        "edgecolor": "xkcd:black",
        "alpha": 1,
        "shape": "rectangle",
    },
    #  States for traffic signals with arrows.
    "LANE_STATE_ARROW_STOP": {
        "facecolor": "xkcd:red",
        "edgecolor": "xkcd:black",
        "alpha": 1,
        "shape": "triangle",
    },
    "LANE_STATE_ARROW_CAUTION": {
        "facecolor": "xkcd:yellow",
        "edgecolor": "xkcd:black",
        "alpha": 1,
        "shape": "triangle",
    },
    "LANE_STATE_ARROW_GO": {
        "facecolor": "xkcd:green",
        "edgecolor": "xkcd:black",
        "alpha": 0.8,
        "shape": "triangle",
    },
    #  Standard round traffic signals.
    "LANE_STATE_STOP": {
        "facecolor": "xkcd:red",
        "edgecolor": "xkcd:black",
        "alpha": 0.8,
        "shape": "circle",
    },
    "LANE_STATE_CAUTION": {
        "facecolor": "xkcd:yellow",
        "edgecolor": "xkcd:black",
        "alpha": 0.8,
        "shape": "circle",
    },
    "LANE_STATE_GO": {
        "facecolor": "xkcd:green",
        "edgecolor": "xkcd:black",
        "alpha": 0.8,
        "shape": "circle",
    },
    #  Flashing light signals.
    "LANE_STATE_FLASHING_STOP": {
        "facecolor": "xkcd:red",
        "edgecolor": "xkcd:black",
        "alpha": 0.5,
        "shape": "rectangle",
    },
    "LANE_STATE_FLASHING_CAUTION": {
        "facecolor": "xkcd:yellow",
        "edgecolor": "xkcd:black",
        "alpha": 0.5,
        "shape": "rectangle",
    },
}

driveway_config = {
    "facecolor": "xkcd:light grey",
    "edgecolor": "xkcd:black",
    "linewidth": 3,
    "alpha": 0.5,
}
