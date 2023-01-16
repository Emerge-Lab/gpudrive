import madrona_python
import gpu_hideseek_python
import torch
import sys
import time
import PIL
import PIL.Image

num_worlds = int(sys.argv[1])
num_steps = int(sys.argv[2])
entities_per_world = int(sys.argv[3])
reset_chance = float(sys.argv[4])

render_width = 64
render_height = 64

sim = gpu_hideseek_python.HideAndSeekSimulator(
        exec_mode = gpu_hideseek_python.ExecMode.CPU,
        gpu_id = 0,
        num_worlds = num_worlds,
        min_entities_per_world = entities_per_world,
        max_entities_per_world = entities_per_world,
        render_width = render_width,
        render_height = render_height,
        debug_compile = False,
)

rgb_observations = sim.rgb_tensor().to_torch()

def dump_rgb(dump_dir, step_idx):
    N = rgb_observations.shape[0]
    A = rgb_observations.shape[1]

    num_wide = min(64, N * A)

    reshaped = rgb_observations.reshape(N * A // num_wide, num_wide, *rgb_observations.shape[2:])
    grid = reshaped.permute(0, 2, 1, 3, 4)

    grid = grid.reshape(N * A // num_wide * render_height, num_wide * render_width, 4)
    grid = grid.type(torch.uint8).cpu().numpy()

    img = PIL.Image.fromarray(grid)
    img.save(f"{dump_dir}/{step_idx}.png", format="PNG")


actions = sim.action_tensor().to_torch()
actions_gpu = torch.zeros_like(actions, device=torch.device('cuda'))

observations_cpu = [
        sim.done_tensor().to_torch(),
        sim.prep_counter_tensor().to_torch(),
        sim.reward_tensor().to_torch(),
        sim.agent_mask_tensor().to_torch(),
        sim.visible_agents_mask_tensor().to_torch(),
        sim.visible_boxes_mask_tensor().to_torch(),
        sim.visible_ramps_mask_tensor().to_torch(),
        sim.agent_data_tensor().to_torch(),
        sim.box_data_tensor().to_torch(),
        sim.ramp_data_tensor().to_torch(),
    ]

observations_gpu = [
        torch.zeros_like(obs, device=torch.device('cuda')) for obs in observations_cpu
    ]

resets = sim.reset_tensor().to_torch()
print(actions.shape, actions.dtype)
print(resets.shape, resets.dtype)
print(rgb_observations.shape, rgb_observations.dtype)

reset_no = torch.zeros_like(resets[:, 0], dtype=torch.int32,
                            device=torch.device('cuda'))
reset_yes = torch.ones_like(resets[:, 0], dtype=torch.int32,
                            device=torch.device('cuda'))
reset_rand = torch.zeros_like(resets[:, 0], dtype=torch.float32,
                              device=torch.device('cuda'))

resets[:, 0] = 1
resets[:, 1] = 3
resets[:, 2] = 2

move_action_slice_gpu = actions_gpu[..., 0:2]
move_action_slice = actions[..., 0:2]

for i in range(5):
    sim.step()

resets[:, 0] = 1

start = time.time()

for i in range(num_steps):
    sim.step()

    for obs_cpu, obs_gpu in zip(observations_cpu, observations_gpu):
        obs_gpu.copy_(obs_cpu)

    torch.rand(reset_rand.shape, out=reset_rand)

    reset_cond = torch.where(reset_rand < reset_chance, reset_yes, reset_no)
    resets[:, 0].copy_(reset_cond)

    torch.randint(-5, 5, move_action_slice.shape,
                  out=move_action_slice_gpu,
                  dtype=torch.int32, device=torch.device('cuda'))
    move_action_slice.copy_(move_action_slice_gpu)

    torch.cuda.synchronize()

    if len(sys.argv) > 5:
        dump_rgb(sys.argv[5], i)

end = time.time()

duration = end - start
print(num_worlds * num_steps / duration, duration)

del sim
