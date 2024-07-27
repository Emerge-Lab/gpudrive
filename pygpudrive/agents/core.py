import torch


def merge_actions(
    actions, actor_ids, reference_actor_shape, verbose=False, device="cuda"
):
    """Combines multiple actor_outputs into one action instance.

    NOTE: Currently assumes that all actions are indices (TODO(dc))

    Args:
        actions (dict of ints): A dictionary of actor outputs.
        actor_ids (dict of ints): A dictionary of actor ids.

    Return:
        torch.Tensor: Full tensor of actions.
    """

    action_tensor = torch.zeros_like(reference_actor_shape).to(device)

    for actor_name in actions.keys():

        if verbose:
            print(
                f"{actor_name} is controlling vehicles: {actor_ids[actor_name]} with actions: {actions[actor_name]}"
            )

        actor_indices = actor_ids[actor_name]
        action_tensor[actor_indices] = actions[actor_name].long()

    return action_tensor
