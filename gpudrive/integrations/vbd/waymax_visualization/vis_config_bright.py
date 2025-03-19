canvas_config = {
    "width": 8,
    "background_color": "xkcd:white",
    "tick_on": False,
}

road_line_config = {
    "Unknown": {
        "color": "xkcd:light grey",
        "linewidth": 1.5,
        "linestyle": "dotted",
        "alpha": 0,
    },
    "BrokenSingleWhite": {
        "color": "xkcd:medium grey",
        "linewidth": 2,
        "linestyle": "--",
        "alpha": 0.5,
    },
    "SolidSingleWhite": {
        "color": "xkcd:medium grey",
        "linewidth": 2,
        "linestyle": "-",
        "alpha": 0.5,
    },
    "SolidDoubleWhite": {
        "color": "xkcd:medium grey",
        "linewidth": 3.5,
        "linestyle": "-",
        "alpha": 0.5,
    },
    "BrokenSingleYellow": {
        "color": "xkcd:yellowish orange",
        "linewidth": 2,
        "linestyle": "--",
        "alpha": 0.5,
    },
    "BrokenDoubleYellow": {
        "color": "xkcd:yellowish orange",
        "linewidth": 3.5,
        "linestyle": "--",
        "alpha": 0.5,
    },
    "SolidSingleYellow": {
        "color": "xkcd:yellowish orange",
        "linewidth": 2,
        "linestyle": "-",
        "alpha": 0.5,
    },
    "SolidDoubleYellow": {
        "color": "xkcd:yellowish orange",
        "linewidth": 2,
        "linestyle": "-",
        "alpha": 0.5,
    },
    "PassingDoubleYellow": {
        "color": "xkcd:yellowish orange",
        "linewidth": 3.5,
        "linestyle": "-.",
        "alpha": 0.5,
    },
}

road_edge_config = {
    "Unknown": {
        "color": "xkcd:brown",
        "linewidth": 2,
        "linestyle": "dotted",
        "alpha": 0.8,
    },
    "Boundary": {
        "color": "xkcd:charcoal",
        "linewidth": 2,
        "linestyle": "-",
        "alpha": 0.8,
    },
    "Median": {
        "color": "xkcd:sage",
        "linewidth": 2,
        "linestyle": "-",
        "alpha": 0.8,
    },
}

lane_config = {
    "TYPE_UNDEFINED": {
        "color": "xkcd:light grey",
        "linewidth": 1.5,
        "linestyle": "dotted",
        "alpha": 1,
    },
    "TYPE_FREEWAY": {
        "color": "xkcd:light blue",
        "linewidth": 1.5,
        "linestyle": "dotted",
        "alpha": 1,
    },
    "TYPE_SURFACE_STREET": {
        "color": "xkcd:light khaki",
        "linewidth": 1.5,
        "linestyle": "dotted",
        "alpha": 1,
    },
    "TYPE_BIKE_LANE": {
        "color": "xkcd:light mint",
        "linewidth": 1.5,
        "linestyle": "dotted",
        "alpha": 1,
    },
}

speed_bump_config = {
    "facecolor": "xkcd:sunflower yellow",
    "edgecolor": "xkcd:black",
    "alpha": 1,
}

crosswalk_config = {
    "facecolor": "None",
    "edgecolor": "xkcd:bluish grey",
    "alpha": 0.2,
}

stop_sign_config = {
    "facecolor": "xkcd:red",
    "edgecolor": "none",
    "linewidth": 1.5,
    "radius": 1.5,
    "alpha": 1,
}

object_config = {
    "TYPE_UNSET": {
        "facecolor": "xkcd:black",
        "linewidth": 1.5,
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
        "linewidth": 1.5,
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
        "linewidth": 1.5,
        "linestyle": "-",
        "alpha": 0.5,
    },
}

signal_config = {
    "linewidth": 0.5,
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
    "linewidth": 1.5,
    "alpha": 0.5,
}
