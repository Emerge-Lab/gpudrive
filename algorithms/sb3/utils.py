import torch

# From stable baselines, adapted np to torch
def explained_variance(
    y_pred: torch.tensor, y_true: torch.tensor
) -> torch.tensor:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = torch.var(y_true)
    return torch.nan if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y
