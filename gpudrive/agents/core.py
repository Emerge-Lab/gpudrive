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



def create_policy_masks(env, num_sim_agents=2):
    policy_mask = torch.zeros_like(env.cont_agent_mask, dtype=torch.int)
    agent_indices = env.cont_agent_mask.nonzero(as_tuple=True)

    for i, (world_idx, agent_idx) in enumerate(zip(*agent_indices)):
        policy_mask[world_idx, agent_idx] = (i % num_sim_agents) + 1

    policy_mask = {f'pi_{int(policy.item())}': (policy_mask == policy)
            for policy in policy_mask.unique() if policy.item() != 0}


    policy_world_mask = {
        world: {f'pi_{p+1}': policy_mask[f'pi_{p+1}'][world] for p in range(num_sim_agents)}
        for world in range(env.cont_agent_mask.shape[0])
    }
    return policy_world_mask