import gpudrive
import pytest

@pytest.fixture(scope="module")
def simulation_results():
    # Create an instance of RewardParams
    reward_params = gpudrive.RewardParams()
    reward_params.rewardType = gpudrive.RewardType.DistanceBased  # Or any other value from the enum
    reward_params.distanceToGoalThreshold = 1.0  # Set appropriate values
    reward_params.distanceToExpertThreshold = 1.0  # Set appropriate values

    # Create an instance of Parameters
    params = gpudrive.Parameters()
    params.polylineReductionThreshold = 0.5  # Set appropriate value
    params.observationRadius = 10.0  # Set appropriate value
    params.collisionBehaviour = gpudrive.CollisionBehaviour.AgentStop  # Set appropriate value
    params.datasetInitOptions = gpudrive.DatasetInitOptions.PadN  # Set appropriate value
    params.rewardParams = reward_params  # Set the rewardParams attribute to the instance created above
    params.maxNumControlledVehicles = 0
    params.IgnoreNonVehicles = True
    params.initAgentsAsStatic = True

    # Now use the 'params' instance when creating SimManager
    sim = gpudrive.SimManager(
        exec_mode=gpudrive.madrona.ExecMode.CPU,
        gpu_id=0,
        num_worlds=1,
        json_path="tests/pytest_data",
        params=params,
        enable_batch_renderer=False, # Optional parameter
        batch_render_view_width=1024,
        batch_render_view_height=1024
    )

    done = sim.done_tensor().to_torch()

    while not done.all():
        sim.step()

    return sim

def test_goal_reaching(simulation_results):
    sim = simulation_results
    info, shape = sim.info_tensor().to_torch(), sim.shape_tensor().to_torch()
    info_valid = info[info[:, :, -1] == float(gpudrive.EntityType.Vehicle)]
    goal_reached = info_valid[:, -2].sum().item()
    assert goal_reached == shape[:, 0].sum().item()
    print("Test passed!")

def test_collision_rate(simulation_results):
    sim = simulation_results
    info = sim.info_tensor().to_torch()

    info_valid = info[info[:, :, -1] == float(gpudrive.EntityType.Vehicle)]
    collisions = info_valid[:, :3].sum().item()
    try:
        assert collisions == 0
    except AssertionError:
        print("Assertion failed! Info tensor:")
        print(info_valid)
        raise
    print("Test passed!")
