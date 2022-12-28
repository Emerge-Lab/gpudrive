import madrona_python
import gpu_hideseek_python
import torch
import sys
import time

num_worlds = int(sys.argv[1])
num_steps = int(sys.argv[2])
entities_per_world = int(sys.argv[3])
reset_chance = float(sys.argv[4])

sim = gpu_hideseek_python.HideAndSeekSimulator(
        gpu_id = 0,
        num_worlds = num_worlds,
        min_entities_per_world = entities_per_world,
        max_entities_per_world = entities_per_world,
        render_width = 64,
        render_height = 64,
        debug_compile = False,
)

actions = sim.move_action_tensor().to_torch()
resets = sim.reset_tensor().to_torch()
rgb_observations = sim.rgb_tensor().to_torch()
print(actions.shape, actions.dtype)
print(resets.shape, resets.dtype)
print(rgb_observations.shape, rgb_observations.dtype)

start = time.time()

reset_no = torch.zeros_like(resets, dtype=torch.int32)
reset_yes = torch.ones_like(resets, dtype=torch.int32)
reset_rand = torch.zeros_like(resets, dtype=torch.float32)

for i in range(num_steps):
    sim.step()

    torch.rand(reset_rand.shape, out=reset_rand)

    reset_cond = torch.where(reset_rand < reset_chance, reset_yes, reset_no)
    resets.copy_(reset_cond)

    #action = get_keyboard_action()
    #
    #if action == -1:
    #    resets[0] = 1
    #elif action == -2:
    #    break
    #else:
    #    actions[0][0] = action

end = time.time()

duration = end - start
print(num_worlds * num_steps / duration, duration)
