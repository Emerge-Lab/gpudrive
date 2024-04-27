import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def read_binary_file(file_name):
    with open(file_name, 'rb') as f:
        # Read the contents of the file into a NumPy array
        events, time_stamps = np.fromfile(f, dtype=np.int64).reshape(2, -1)
        # set the time stamp of first event to be 0
        time_stamps = [i - time_stamps[0] for i in time_stamps]
    return events, time_stamps


# apt to change
event_dict = {0: 1, 2: 3, 4: 5}
color_dict = {1: 'r', 3: 'g', 5: 'b'}


def get_steps(events, time_stamps):
    steps = []
    for i in range(len(events)):
        if events[i:i + 4].tolist() == [2, 3, 4, 5]:
            steps.append(time_stamps[i:i + 4])
    return steps


def plot_events(events,
                time_stamps,
                file_name,
                exclude_init=True,
                drop_warmup=2,
                display_steps=20):

    steps = get_steps(events, time_stamps)
    steps = steps[drop_warmup:drop_warmup + display_steps]
    steps = [[i - s[0] for i in s] for s in steps]

    max_time = max(max(s) for s in steps)

    num_steps = len(steps)
    x_limit = 200
    pixel_per_step = 10
    bar_interval = 2
    y_limit = num_steps * (pixel_per_step + bar_interval)

    img = Image.new("RGB", (x_limit, y_limit), "white")
    draw = ImageDraw.Draw(img)

    def cast_coor(timestamp, limit=x_limit):
        assert timestamp <= max_time
        return int(timestamp / max_time * limit)

    for i, s in enumerate(steps):
        s = [i - s[0] for i in s]
        y = pixel_per_step / 2 + i * (pixel_per_step + bar_interval)
        draw.line((0, y, cast_coor(s[1]), y),
                  fill="green",
                  width=pixel_per_step)
        draw.line((cast_coor(s[2]), y, cast_coor(s[3]), y),
                  fill="blue",
                  width=pixel_per_step)
        # print(s[1]/1000000, (s[3] - s[2])/1000000, s[1] / (s[3] - s[2] + s[1]) )

    img.save(file_name + '_events.png')

    # num_events = len(events) // 2
    # event_stack = [0]
    # event_starts = []
    # event_ends = []
    # event_colors = []
    # for i in range(1, len(events)):
    #     if events[i] in event_dict:
    #         event_stack.append(i)
    #     else:
    #         assert (events[i] == event_dict[events[event_stack[-1]]])
    #         if exclude_init and events[i] == 1:
    #             num_events -= 1
    #             continue
    #         event_starts.append(time_stamps[event_stack[-1]])
    #         event_ends.append(time_stamps[i])
    #         event_colors.append(color_dict[events[i]])
    #         event_stack.pop()
    # assert num_events == len(event_starts) == len(event_ends)

    # num_events -= drop_warmup
    # event_starts = event_starts[drop_warmup:]
    # event_ends = event_ends[drop_warmup:]
    # event_colors = event_colors[drop_warmup:]

    # plt.barh(range(num_events),
    #          [event_ends[i] - event_starts[i] for i in range(num_events)],
    #          left=event_starts,
    #          color=event_colors)
    # plt.xlim(min(event_starts), max(event_ends))
    # plt.savefig(file_name + '_events.png')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("python parse_tracing.py [file_name]")
        exit()

    events, time_stamps = read_binary_file(sys.argv[1])
    plot_events(events, time_stamps, sys.argv[1])