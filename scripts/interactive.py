import madrona_python
import gpu_hideseek_python
import torch
import torchvision
import sys
import termios
import tty

def get_single_char():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(sys.stdin.fileno())
    
    ch = sys.stdin.read(1)
    
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return ch

def get_keyboard_action():
    while True:
        key_action = get_single_char()

        if key_action == 'w':
            return 1
        elif key_action == 'a':
            return 2
        elif key_action == 'd':
            return 3
        elif key_action == 's':
            return  4
        elif key_action == 'c':
            return  5
        elif key_action == 'z':
            return  6
        elif key_action == 'r':
            return -1
        elif key_action == 'q':
            return -2
        elif key_action == ' ':
            return 0
        else:
            continue

sim = gpu_hideseek_python.HideAndSeekSimulator(
        exec_mode = gpu_hideseek_python.ExecMode.CUDA,
        gpu_id = 0,
        num_worlds = 1,
        min_entities_per_world = 50,
        max_entities_per_world = 50,
        render_width = 1024,
        render_height = 1024
)

actions = sim.move_action_tensor().to_torch()
resets = sim.reset_tensor().to_torch()
rgb_observations = sim.rgb_tensor().to_torch()
print(actions.shape, actions.dtype)
print(resets.shape, resets.dtype)
print(rgb_observations.shape, rgb_observations.dtype)

while True:
    sim.step()
    torchvision.utils.save_image((rgb_observations[0].float() / 255).permute(2, 0, 1), sys.argv[1])

    action = get_keyboard_action()
    
    if action == -1:
        resets[0] = 1
    elif action == -2:
        break
    else:
        actions[0][0] = action

del sim
