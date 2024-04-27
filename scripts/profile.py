import os
import sys
import json
import subprocess
import pandas as pd

NUM_THREADS_PER_BLOCK = 256
NUM_SMS = 82
BASE_STEP = 10
DIR_PATH = "/tmp/profile_blocks__megakernel_events"
# only take action when the change of config for certain node can contribute to at least 0.1% overall acceleration
THRESHOLD = 1000


def profile_madrona(bench_cmd,
                    block_config=range(1, 7),
                    cache="/tmp/madcache"):
    try:
        #os.remove(cache)
        pass
    except:
        pass

    for config in block_config:
        profile_command = "MADRONA_MWGPU_ENABLE_PGO=1 MADRONA_MWGPU_TRACE_NAME=profile_{block}_block MADRONA_MWGPU_EXEC_CONFIG_OVERRIDE={thread},{block},{sm} MADRONA_MWGPU_KERNEL_CACHE={cache} {bench_cmd}".format(
            thread=NUM_THREADS_PER_BLOCK,
            block=config,
            sm=NUM_SMS,
            cache=cache,
            bench_cmd=bench_cmd)
        subprocess.run(profile_command, shell=True, text=True)


def parse_traces(block_config=range(1, 7)):
    from parse_device_tracing import parse_device_logs, step_analysis

    tabular_data = [
        pd.DataFrame({
            "nodeID": [],
            "funcID": [],
            "duration (ns)": [],
            "invocations": [],
            "percentage (%)": [],
            "SM utilization": []
        }) for _ in block_config
    ]

    isExist = os.path.exists(DIR_PATH)
    if not isExist:
        os.mkdir(DIR_PATH)
    for i in block_config:
        path_to_trace = "/tmp/profile_{}_block_madrona_device_tracing.bin".format(
            i)

        with open(path_to_trace, 'rb') as f:
            events = bytearray(f.read())
            assert len(events) % 40 == 0
            print("{} events were logged in total".format(len(events) // 40))
            log_steps = parse_device_logs(events)
        assert len(log_steps) > BASE_STEP

        with pd.ExcelWriter(DIR_PATH +
                            "/block_{}_metrics.xlsx".format(i)) as writer:
            step_analysis(log_steps[BASE_STEP],
                          DIR_PATH + "/block_{}.png".format(i),
                          tabular_data[i - 1]).to_excel(writer, index=False)


def generate_json(block_config=range(1, 7)):
    tabular_data = [
        pd.read_excel(DIR_PATH + "/block_{}_metrics.xlsx".format(i))
        for i in block_config
    ]
    tabular_data = ["paddding"] + tabular_data

    overall_durations = {
        i: sum(tabular_data[i]['duration (ns)']) for i in block_config
    }
    base_config, base_duration = min(overall_durations.items(),
                                     key=lambda x: x[1])

    duration_deduction = 0
    split_nodes = {}
    for i, node in enumerate(tabular_data[base_config]['nodeID']):
        min_duration = tabular_data[base_config]['duration (ns)'][i]
        for b in block_config:
            if min_duration - tabular_data[b]['duration (ns)'][
                    i] > base_duration / THRESHOLD:
                duration_deduction += min_duration - tabular_data[b][
                    'duration (ns)'][i]
                min_duration = tabular_data[b]['duration (ns)'][i]
                split_nodes[node] = b

    print(
        "with default {} block per SM config and following exceptions:".format(
            base_config), split_nodes)
    print("the estimated acceleration will be {:.3f}%".format(
        duration_deduction / base_duration * 100))

    with open(DIR_PATH + '/node_blocks.json', 'w') as f:
        json.dump(split_nodes, f)

if len(sys.argv) < 2:
    print(f"{sys.argv[0]} benchmark command and args", file=sys.stderr)
    sys.exit(1)

bench_cmd = " ".join(sys.argv[1:])

profile_madrona(bench_cmd)
parse_traces()
generate_json()