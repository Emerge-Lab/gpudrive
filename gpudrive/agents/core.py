import torch


def merge_actions(
    actor_actions_dict,
    actor_ids_dict,
    reference_action_tensor,
    verbose=False,
    device="cuda",
):
    """Combines multiple actor_outputs into one action instance.

    Args:
        actions (dict of ints): A dictionary of actor outputs.
        actor_ids (dict of ints): A dictionary of actor ids.
        reference_action_tensor (torch.Tensor): A reference tensor of shape (num_worlds, max_num_controllable_agents) to map the actor outputs to.

    Return:
        torch.Tensor: Tensor of shape (num_worlds, max_num_controllable_agents) filled with actor actions.
    """

    action_tensor = (
        torch.zeros(reference_action_tensor.shape)
        .type(torch.LongTensor)
        .to(device)
    )

    for actor_name in actor_actions_dict.keys():
        for world_idx in range(len(actor_ids_dict[actor_name])):
            if verbose:
                print(
                    f"{actor_name} is controlling vehicles: {actor_ids_dict[actor_name][world_idx]} in world {world_idx} \n with actions: {actor_ids_dict[actor_name]}"
                )
            actor_indices_in_world = actor_ids_dict[actor_name][world_idx]
            action_tensor[
                world_idx, actor_indices_in_world
            ] = actor_actions_dict[actor_name][world_idx].long()

    return action_tensor
