import madrona_gpudrive
import pytest
import torch

@pytest.fixture(scope="module")
def sim_init():
    reward_params = madrona_gpudrive.RewardParams()
    reward_params.rewardType = madrona_gpudrive.RewardType.DistanceBased 
    reward_params.distanceToGoalThreshold = 1.0 
    reward_params.distanceToExpertThreshold = 1.0  

    params = madrona_gpudrive.Parameters()
    params.polylineReductionThreshold = 0.5  
    params.observationRadius = 10.0 
    params.collisionBehaviour = madrona_gpudrive.CollisionBehaviour.AgentStop
    params.rewardParams = reward_params 
    params.maxNumControlledAgents = 2 # we are going to use the second vehicle as the controlled vehicle
    params.IgnoreNonVehicles = True
    params.dynamicsModel = madrona_gpudrive.DynamicsModel.DeltaLocal

    sim = madrona_gpudrive.SimManager(
        exec_mode=madrona_gpudrive.madrona.ExecMode.CPU,
        gpu_id=0,
        scenes=["tests/pytest_data/test.json"],
        params=params
    )

    return sim

def test_forward_inverse_delta_dynamics(sim_init):
    sim = sim_init
    
    valid_agent_idx = 1
    done_tensor = sim.done_tensor().to_torch()
    expert_trajectory_tensor = sim.expert_trajectory_tensor().to_torch()
    action_tensor = sim.action_tensor().to_torch()
    absolute_obs = sim.absolute_self_observation_tensor().to_torch()
    self_obs = sim.self_observation_tensor().to_torch()
    actions = torch.zeros_like(action_tensor)

    traj = expert_trajectory_tensor[:,valid_agent_idx].squeeze()
    pos = traj[:2*91].view(91,2)
    vel = traj[2*91:4*91].view(91,2)
    headings = traj[4*91:5*91].view(91,1)
    invActions = traj[6*91:16*91].view(91,10)
    print('invActions', invActions[:2])
    position = absolute_obs[0,valid_agent_idx,:2]
    heading = absolute_obs[0,valid_agent_idx,7]
    speed = self_obs[0, valid_agent_idx, 0]

    assert torch.allclose(position, pos[0], atol=1e-2), f"Position mismatch: {position} vs {pos[0]}"
    assert pytest.approx(heading.item(), abs=1e-2) == headings[0].item(), f"Heading mismatch: {heading.item()} vs {headings[0].item()}"
    assert pytest.approx(speed.item(), abs=1e-2) == torch.norm(vel[0]).item(), f"Speed mismatch: {speed.item()} vs {torch.norm(vel[0]).item()}"

    actions[:,valid_agent_idx,:3] = invActions[0,:3]
    action_tensor.copy_(actions)
    sim.step()
    
    assert torch.allclose(position, pos[1], atol=2e-2), f"Position mismatch: {position} vs {pos[1]} at step {1}"
    assert pytest.approx(heading.item(), abs=3e-3) == headings[1].item(), f"Heading mismatch: {heading.item()} vs {headings[1].item()}"
    assert pytest.approx(speed.item(), abs=1e-3) == torch.norm(vel[1]).item(), f"Speed mismatch: {speed.item()} vs {torch.norm(vel[1]).item()}"