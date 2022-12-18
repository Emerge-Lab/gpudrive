import madrona_python
import gpu_hideseek_python
import torch
import sys
import time

num_worlds = int(sys.argv[1])
num_steps = int(sys.argv[2])

sim = gpu_hideseek_python.HideAndSeekSimulator(
        gpu_id = 0,
        num_worlds = num_worlds,
        render_width = 64,
        render_height = 64,
        debug_compile = True,
)

actions = sim.move_action_tensor().to_torch()
resets = sim.reset_tensor().to_torch()
rgb_observations = sim.rgb_tensor().to_torch()
print(actions.shape, actions.dtype)
print(resets.shape, resets.dtype)
print(rgb_observations.shape, rgb_observations.dtype)

start = time.time()

for i in range(num_steps):
    sim.step()

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
print(num_steps / duration, duration)
