import os
import sys
import argparse
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

HIDE_SEEK = False
SPLIT_LINES = False
MAX_BLOCKS_PER_SM = 6


def parse_device_logs(events):
    LOG_STEPS = {}
    STEP = -1

    def new_step(num_warps, num_blocks, num_sms):
        nonlocal STEP
        STEP += 1
        LOG_STEPS[STEP] = {
            "events": {},
            "SMs": {},
            "mapping": {},
            # "final_cycles": {},
            "start_timestamp": 0xFFFFFFFFFFFFFFFF,
            "final_timestamp": 0,
            "num_warps": num_warps,
            "num_blocks": num_blocks,
            "num_sms": num_sms,
            "configs": {}
        }

    for i in tqdm(range(0, len(events), 40)):
        array = events[i:i + 40]

        event = int.from_bytes(array[:4], byteorder='little')
        funcID = int.from_bytes(array[4:8], byteorder='little')
        numInvocations = int.from_bytes(array[8:12], byteorder='little')
        nodeID = int.from_bytes(array[12:16], byteorder='little')
        warpID = int.from_bytes(array[16:20], byteorder='little')
        blockID = int.from_bytes(array[20:24], byteorder='little')
        smID = int.from_bytes(array[24:28], byteorder='little')
        logIndex = int.from_bytes(array[28:32], byteorder='little')
        cycleCount = int.from_bytes(array[32:40], byteorder='little')

        if STEP != -1:
            # to make a unique warp id
            warpID += blockID * LOG_STEPS[STEP]["num_warps"]

        # print("event: {}, funcID: {}, numInvocations: {}, nodeID: {}, warpID: {}, blockID: {}, smID: {}, cycleCount: {}, logIndex: {}".format(event, funcID, numInvocations, nodeID, warpID, blockID, smID, cycleCount, logIndex))
        if event == 0:
            # for calibration event, we log kernel config instead of node
            # when logIndex == 0, it is the first node of the whole step
            if logIndex == 0:
                new_step(num_warps=funcID,
                         num_blocks=numInvocations,
                         num_sms=nodeID)
                LOG_STEPS[STEP]["start_timestamp"] = cycleCount
            else:
                assert LOG_STEPS[STEP]["num_warps"] == funcID and LOG_STEPS[
                    STEP]["num_sms"] == nodeID
                LOG_STEPS[STEP]["num_blocks"] = numInvocations

        elif event in [1, 2]:
            if nodeID not in LOG_STEPS[STEP]["mapping"]:
                LOG_STEPS[STEP]["mapping"][nodeID] = (funcID, numInvocations)
            else:
                assert LOG_STEPS[STEP]["mapping"][nodeID] == (funcID,
                                                              numInvocations)

            if nodeID not in LOG_STEPS[STEP]["events"]:
                LOG_STEPS[STEP]["events"][nodeID] = {
                    event: (smID, warpID, cycleCount)
                }
            else:
                assert event not in LOG_STEPS[STEP]["events"][nodeID]
                LOG_STEPS[STEP]["events"][nodeID][event] = (smID, warpID,
                                                            cycleCount)
            if nodeID not in LOG_STEPS[STEP]["configs"]:
                LOG_STEPS[STEP]["configs"][nodeID] = {
                    "num_blocks": LOG_STEPS[STEP]["num_blocks"]
                }

        elif event in [3, 4]:
            if smID not in LOG_STEPS[STEP]["SMs"]:
                assert event == 3
                LOG_STEPS[STEP]["SMs"][smID] = {
                    (numInvocations, nodeID, warpID): [cycleCount]
                }
            else:
                if (numInvocations, nodeID,
                        warpID) in LOG_STEPS[STEP]["SMs"][smID]:
                    assert event == 4
                    LOG_STEPS[STEP]["SMs"][smID][(numInvocations, nodeID,
                                                  warpID)].append(cycleCount)
                else:
                    assert event == 3
                    LOG_STEPS[STEP]["SMs"][smID][(numInvocations, nodeID,
                                                  warpID)] = [cycleCount]

        elif event == 5:
            # assert warpID not in LOG_STEPS[STEP]["final_cycles"]
            # LOG_STEPS[STEP]["final_cycles"][warpID] = cycleCount
            LOG_STEPS[STEP]["final_timestamp"] = max(
                LOG_STEPS[STEP]["final_timestamp"], cycleCount)
        else:
            assert (False & "event {} not supported".format(event))

    # drop the last step which might be corrupted
    # del LOG_STEPS[STEP]
    print("At the end, complete traces for {} steps are generated".format(STEP))

    for s in LOG_STEPS:
        LOG_STEPS[s]["final_timestamp"] -= LOG_STEPS[s]["start_timestamp"]
        # for b in LOG_STEPS[s]["final_cycles"]:
        #     LOG_STEPS[s]["final_cycles"][b] -= LOG_STEPS[s]["start_timestamp"]

    return LOG_STEPS


def serialized_analysis(step_log, nodes_map):

    def calibrate(timestamp, reference):
        return timestamp[2] - reference

    for i in range(max(step_log["events"]) + 1):
        # skipped nodes
        if i not in step_log["events"]:
            continue
        nodes_map[i] = {
            "nodeID":
                i,
            "funcID":
                step_log["mapping"][i][0],
            "invocations":
                step_log["mapping"][i][1],
            "start":
                calibrate(step_log["events"][i][1],
                          step_log["start_timestamp"]),
            "end":
                calibrate(step_log["events"][i][2],
                          step_log["start_timestamp"]),
            "SM utilization": []
        }
        nodes_map[i][
            "duration (ns)"] = nodes_map[i]["end"] - nodes_map[i]["start"]

    total_exec_time = sum(v["duration (ns)"] for v in nodes_map.values())
    for i in nodes_map:
        nodes_map[i]["percentage (%)"] = nodes_map[i][
            "duration (ns)"] / total_exec_time * 100


def block_analysis(step_log, nodes_map):
    sm_execution = {k: [] for k in step_log["SMs"].keys()}
    block_exec_time = {
        "blocks": {k: {} for k in step_log["SMs"].keys()},  # horizontal
        "nodes": {k: {} for k in step_log["SMs"].keys()},  # vertical
    }

    for sm in step_log["SMs"]:
        for (_, nodeID, warpID), time_stamps in step_log["SMs"][sm].items():
            assert (len(time_stamps) == 2)
            start = time_stamps[0]
            end = max(time_stamps[1:])
            # confirm clock does proceed within an SM
            assert end >= start
            assert start > step_log["start_timestamp"]

            start, end = [i - step_log["start_timestamp"] for i in [start, end]]
            sm_execution[sm].append((start, end))

            if warpID not in block_exec_time["blocks"][sm]:
                block_exec_time["blocks"][sm][warpID] = [(start, end, nodeID)]
            else:
                assert start > block_exec_time["blocks"][sm][warpID][-1][1]
                block_exec_time["blocks"][sm][warpID].append(
                    (start, end, nodeID))

            if nodeID not in block_exec_time["nodes"][sm]:
                block_exec_time["nodes"][sm][nodeID] = []
            block_exec_time["nodes"][sm][nodeID].append((start, end, warpID))

        block_exec_time["blocks"][sm] = {
            k: sorted(v) for k, v in block_exec_time["blocks"][sm].items()
        }
        block_exec_time["nodes"][sm] = {
            k: sorted(v) for k, v in block_exec_time["nodes"][sm].items()
        }

    for s in sm_execution:
        intervals = []
        for start, end in sm_execution[s]:
            if len(intervals) == 0:
                intervals.append((start, end))
                continue
            p = 0
            while p < len(intervals):
                if start > intervals[p][1]:
                    p += 1
                    continue
                elif end < intervals[p][0]:
                    intervals.insert(p, (start, end))
                    break
                else:
                    intervals[p] = (min(start, intervals[p][0]),
                                    max(end, intervals[p][1]))
                    break
            else:
                intervals.insert(p, (start, end))

        i_pointer = 0
        for k, v in nodes_map.items():
            occupied_time = 0
            while i_pointer < len(intervals):
                i_start, i_end = intervals[i_pointer]
                if v["end"] < i_start:
                    break
                elif v["start"] <= i_start and v["end"] >= i_end:
                    occupied_time += i_end - i_start
                    i_pointer += 1
                    continue
                else:
                    assert (False and "no intersections")
            nodes_map[k]["SM utilization"].append(occupied_time /
                                                  nodes_map[k]["duration (ns)"])

    for i in nodes_map:
        assert (len(nodes_map[i]["SM utilization"]) == len(sm_execution))
        nodes_map[i]["SM utilization"] = sum(
            nodes_map[i]["SM utilization"]) / len(sm_execution)

    print(
        "For each SM on average, {:.3f}% of the time there is at least one block is running on"
        .format(
            sum(v["SM utilization"] * v["percentage (%)"]
                for v in nodes_map.values())))

    return block_exec_time


COLORS = [
    "blue", "orange", "red", "green", "purple", "cyan", "pink", "magenta",
    "olive", "navy", "teal", "maroon", "yellow", "black"
]


def plot_events(step_log, nodes_map, blocks, file_name, args):
    # todo: here we have an assumption that each SM has the same number of blocks, which are expected to be true
    num_sms = step_log["num_sms"]
    # num_block_per_sm = step_log["num_blocks"]
    num_warp_per_sm = step_log["num_warps"] * MAX_BLOCKS_PER_SM
    # num_block_per_sm = 8
    num_pixel_per_warp = 1
    sm_interval_pixel = num_pixel_per_warp * 3
    num_pixel_per_sm = num_warp_per_sm * num_pixel_per_warp + sm_interval_pixel
    y_blank = num_pixel_per_warp * num_warp_per_sm * (num_sms // 8)
    y_limit = num_sms * num_pixel_per_sm + y_blank
    x_limit = y_limit * args.aspect_ratio
    print("the figure size will be {}x{}".format(x_limit, y_limit))

    top_nodes = sorted([
        i[0] for i in sorted(nodes_map.items(),
                             key=lambda item: item[1]["duration (ns)"])
        [len(nodes_map) - args.num_highlight_nodes:]
    ])

    colors = {}
    if HIDE_SEEK:
        colors = {
            # narrowphase
            28: (0, 76, 153),
            # solvers
            20: (0, 102, 204),
            22: (0, 128, 255),
            24: (102, 178, 255),
            26: (178, 216, 255),
            # broadphase
            30: (199, 31, 102),
            34: (230, 96, 152),
            36: (248, 181, 209),
            # Collect Observations System
            52: (139, 235, 219),
            # Compute Visibility System
            54: (90, 184, 168),
            # LIDAR System
            56: (148, 112, 206)
        }
    else:
        for n in top_nodes:
            func = nodes_map[n]["funcID"]
            if func not in colors and len(colors) < len(COLORS):
                colors[func] = COLORS[len(colors)]
    print("Color mapping for functions:", colors)

    img = Image.new("RGB", (x_limit, y_limit), "white")
    draw = ImageDraw.Draw(img)

    def cast_coor(timestamp, limit=x_limit):
        assert (timestamp <= step_log["final_timestamp"])
        if args.fixed_scale:
            return int(timestamp / args.tpp)
        else:
            return int(timestamp / step_log["final_timestamp"] * limit)

    color_span = {}
    for n in nodes_map:
        if nodes_map[n]["funcID"] not in colors:
            continue
        left, right = cast_coor(nodes_map[n]["start"]), cast_coor(
            nodes_map[n]["end"])
        color_span[(left, right)] = colors[nodes_map[n]["funcID"]]
    sorted_color_span = sorted(color_span.keys())

    num_stamps = active_warps = 0

    for sm, b in blocks.items():
        node_pixels_sm = {}
        y = (sm + 1) * num_pixel_per_sm
        vertical_pixels = {}
        for warp, events in b.items():
            # to avoid duplication
            last_end_pixel = -1
            for e in events:
                start = max(last_end_pixel + 1, cast_coor(e[0]))
                end = cast_coor(e[1])
                if e[2] not in node_pixels_sm:
                    node_pixels_sm[e[2]] = [start, end]
                else:
                    node_pixels_sm[e[2]] = [
                        min(node_pixels_sm[e[2]][0], start),
                        max(node_pixels_sm[e[2]][1], end)
                    ]
                for i in range(start, end + 1):
                    if i not in vertical_pixels:
                        vertical_pixels[i] = 1 * MAX_BLOCKS_PER_SM / step_log[
                            "configs"][e[2]]["num_blocks"]
                    else:
                        vertical_pixels[i] += 1 * MAX_BLOCKS_PER_SM / step_log[
                            "configs"][e[2]]["num_blocks"]
                last_end_pixel = end

        n_pointer = 0
        for node_start, node_end in sorted(node_pixels_sm.values()):
            for p in range(node_start, node_end + 1):
                if p not in vertical_pixels:
                    vertical_pixels[p] = 0
                num_stamps += num_warp_per_sm
                active_warps += vertical_pixels[p]

                while n_pointer < len(color_span):
                    left, right = sorted_color_span[n_pointer]
                    if p > right:
                        n_pointer += 1
                        continue
                    elif p < left:
                        bar_color = (0, 0, 0)
                        break
                    else:
                        bar_color = color_span[(left, right)]
                    break
                else:
                    bar_color = (0, 0, 0)
                assert vertical_pixels[p] <= num_warp_per_sm
                # if vertical_pixels[p] != 0:
                draw.line(
                    (p, y, p, y - int(vertical_pixels[p] * num_pixel_per_warp)),
                    fill=bar_color,
                    width=1 if vertical_pixels[p] != 0 else 0)
                y_low = y - int(vertical_pixels[p] * num_pixel_per_warp)
                y_low -= 1 if vertical_pixels[p] != 0 else 0
                y_high = y - num_warp_per_sm * num_pixel_per_warp
                if y_low <= y_high:
                    pass
                else:
                    draw.line((p, y_low, p, y_high),
                              fill=(211, 211, 211),
                              width=1)

    if SPLIT_LINES:
        # indicate the start of each splitted kernel
        nodes = sorted(nodes_map.keys())
        last_node = nodes[0]
        for n in nodes[1:]:
            if step_log["configs"][n]["num_blocks"] != step_log["configs"][
                    last_node]["num_blocks"]:
                node_start = cast_coor(nodes_map[n]["start"])
                draw.line((node_start, 0, node_start, y_limit - y_blank / 2),
                        fill="plum",
                        width=2)
            last_node = n

    print("Percentage of active warps is {:.2f}%".format(active_warps /
                                                         num_stamps * 100))

    fontsize = 65
    font = ImageFont.load_default(size=fontsize)

    if not HIDE_SEEK:
        # mark the start and the end of major nodes_map
        y_shift = 0.9
        for n in top_nodes:
            # for n, v in nodes_map.items():
            left, right = cast_coor(nodes_map[n]["start"]), cast_coor(
                nodes_map[n]["end"])
            draw.line((left, 0, left, y_limit), fill="red", width=1)
            draw.line((right, 0, right, y_limit), fill="green", width=1)
            draw.text((left, y_limit - y_blank * y_shift),
                      " f: {}\n t: {:.3f}ms\n {:.1f}%".format(
                          nodes_map[n]["funcID"],
                          (nodes_map[n]["duration (ns)"]) / 1000000,
                          nodes_map[n]["percentage (%)"]),
                      fill=(0, 0, 0), font = font)
            y_shift = 1.3 - y_shift

    img.save(file_name)


def step_analysis(step_log, file_name, tabular_data, args=None):
    nodes_map = {}
    serialized_analysis(step_log, nodes_map)

    block_exec_time = block_analysis(step_log, nodes_map)
    if args is not None:
        plot_events(step_log, nodes_map, block_exec_time["blocks"], file_name, args)

    for n in nodes_map:
        tabular_data = pd.concat([
            tabular_data,
            pd.DataFrame({k: [v] for k, v in nodes_map[n].items()})
        ])
    return tabular_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_file", type=str, required=True)
    parser.add_argument("--start_step",
                        type=int,
                        default=10,
                        help="analysis start from which step")
    parser.add_argument("--num_steps",
                        type=int,
                        default=5,
                        help="number of steps to be analyzed")
    parser.add_argument("--num_highlight_nodes", type=int, default=16)
    parser.add_argument("--aspect_ratio", type=float, default=2)
    parser.add_argument("--fixed_scale", action="store_true")
    parser.add_argument("--tpp",
                        type=int,
                        default=8000,
                        help="time(ns) per pixel")

    args = parser.parse_args()

    with open(args.trace_file, 'rb') as f:
        events = bytearray(f.read())
        assert len(events) % 40 == 0
        print("{} events were logged in total".format(len(events) // 40))
        LOG_STEPS = parse_device_logs(events)

    steps = args.num_steps
    start_from = args.start_step

    dir_path = args.trace_file + "_megakernel_events"
    isExist = os.path.exists(dir_path)
    if not isExist:
        os.mkdir(dir_path)
    # todo: limit
    assert start_from < len(LOG_STEPS)
    end_at = min(start_from + steps, len(LOG_STEPS))

    tabular_data = [
        pd.DataFrame({
            "nodeID": [],
            "funcID": [],
            "duration (ns)": [],
            "invocations": [],
            "percentage (%)": [],
            "SM utilization": []
        }) for _ in tqdm(range(start_from, end_at))
    ]

    with pd.ExcelWriter(dir_path + "/metrics.xlsx") as writer:
        for s in tqdm(range(start_from, end_at)):
            step_analysis(LOG_STEPS[s], dir_path + "/step{}.png".format(s),
                          tabular_data[s - start_from],
                          args).to_excel(writer,
                                         sheet_name="step{}".format(s),
                                         index=False)