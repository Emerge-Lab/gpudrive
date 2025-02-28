import torch


def merge_actions(
    actor_actions_dict,
    reference_action_tensor,
    policy_masks,
    verbose=False,
    device="cuda"
    
):
    """Combines multiple actor_outputs into one action instance.

    Args:
        actions (dict of ints): A dictionary of actor outputs.
        actor_ids (dict of ints): A dictionary of actor ids.
        reference_action_tensor (torch.Tensor): A reference tensor of shape (num_worlds, max_num_controllable_agents) to map the actor outputs to.

    Return:
        torch.Tensor: Tensor of shape (num_worlds, max_num_controllable_agents) filled with actor actions.
    """

    action_tensor = torch.zeros_like(reference_action_tensor, dtype=torch.long, device=device)
    
    count = 0
    for world_idx, world in enumerate(reference_action_tensor):
        for agent_idx, agent in enumerate(world):
            if agent:
                policy_mask = policy_masks[world_idx]
                for policy_name, mask in policy_mask.items():
                    if mask[agent_idx]:
                        actions = actor_actions_dict.get(policy_name, [])
                        if count < len(actions):
                            action_tensor[world_idx, agent_idx] = actions[count]
                            count += 1
                            break
    
    return action_tensor
