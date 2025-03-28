import torch
import numpy as np

# Parameters
MAX_NUM_OBJECTS = 32
MAX_MAP_POINTS = 3000
MAX_POLYLINES = 256
MAX_TRAFFIC_LIGHTS = 16
num_points_polyline = 30


def duplicate_batch(batch: dict, num_samples: int):
    """Duplicates the batch for the given number of samples."""
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert value.shape[0] == 1, "Only support batch size of 1"
            batch[key] = torch.cat([value] * num_samples, dim=0)

    return batch


def torch_dict_to_numpy(input: dict):
    output = {}
    for key, value in input.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.detach().cpu().numpy()
        else:
            output[key] = value
    return output


def stack_dict(input: list):
    list_len = len(input)
    if list_len == 0:
        return {}
    key_to_list = {}
    for key in input[0].keys():
        key_to_list[key] = [input[i][key] for i in range(list_len)]

    output = {}
    for key, value in key_to_list.items():
        if isinstance(value[0], np.ndarray):
            output[key] = np.stack(value, axis=0)
        elif isinstance(value[0], dict):
            output[key] = stack_dict(value)
        else:
            output[key] = value

    return output
