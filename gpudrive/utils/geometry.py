def restore_mean(x, y, mean_x, mean_y):
    """
    In GPUDrive, everything is centered at zero by subtracting the mean.
    This function reapplies the mean to go back to the original coordinates.
    The mean (xyz) is exported per world as world_means_tensor.
    Args:
        x (torch.Tensor): x coordinates
        y (torch.Tensor): y coordinates
        mean_x (torch.Tensor): mean of x coordinates. Shape: (num_worlds, 1)
        mean_y (torch.Tensor): mean of y coordinates. Shape: (num_worlds, 1)
    """
    return x + mean_x, y + mean_y


def normalize_min_max(tensor, min_val, max_val):
    """Normalizes an array of values to the range [-1, 1].

    Args:
        x (np.array): Array of values to normalize.
        min_val (float): Minimum value for normalization.
        max_val (float): Maximum value for normalization.

    Returns:
        np.array: Normalized array of values.
    """
    return 2 * ((tensor - min_val) / (max_val - min_val)) - 1


def normalize_min_max_inplace(tensor, min_val, max_val):
    """Normalizes an array of values to the range [-1, 1].
    Args:
        x (np.array): Array of values to normalize.
        min_val (float): Minimum value for normalization.
        max_val (float): Maximum value for normalization.
    """
    tensor.sub_(min_val).div_(max_val - min_val).mul_(2).sub_(1)