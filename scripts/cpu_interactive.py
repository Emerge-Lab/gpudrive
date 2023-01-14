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

class Action:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.r = 0
        self.g = 0
        self.l = 0
        self.reset = 0

def get_keyboard_action():
    while True:
        key_action = get_single_char()

        result = Action()

        if key_action == 'w':
            result.y = 5
        elif key_action == 'a':
            result.x = -5
        elif key_action == 'd':
            result.x = 5
        elif key_action == 's':
            result.y = -5
        elif key_action == 'q':
            result.r = 1
        elif key_action == 'e':
            result.r = -1
        elif key_action == 'g':
            result.g = 1
        elif key_action == 'l':
            result.l = 1
        elif key_action == 'x':
            result.reset = -1
        elif key_action == '1':
            result.reset = 1
        elif key_action == '2':
            result.reset = 2
        elif key_action == '3':
            result.reset = 3
        elif key_action == '4':
            result.reset = 4
        elif key_action == '5':
            result.reset = 5
        elif key_action == '6':
            result.reset = 6
        elif key_action == ' ':
            pass
        else:
            continue

        return result

sim = gpu_hideseek_python.HideAndSeekSimulator(
        exec_mode = gpu_hideseek_python.ExecMode.CPU,
        gpu_id = 0,
        num_worlds = 1,
        min_entities_per_world = 5,
        max_entities_per_world = 5,
        render_width = 1024,
        render_height = 1024,
        lidar_render = True,
        debug_compile = False,
)

for i in range(100):
    sim.step()

#actions = sim.action_tensor().to_torch()
#rewards = sim.reward_tensor().to_torch()
#agent_valid_masks = sim.agent_mask_tensor().to_torch()
#agent_visibility_masks = sim.visible_agents_mask_tensor().to_torch()
#box_visibility_masks = sim.visible_boxes_mask_tensor().to_torch()
#ramp_visibility_masks = sim.visible_ramps_mask_tensor().to_torch()
#agent_data = sim.agent_data_tensor().to_torch()
#resets = sim.reset_tensor().to_torch()
#prep_counter = sim.prep_counter_tensor().to_torch()
#dones = sim.done_tensor().to_torch()
#
#rgb_observations = sim.rgb_tensor().to_torch()
#print(actions.shape, actions.dtype)
#print(resets.shape, resets.dtype)
#print(rgb_observations.shape, rgb_observations.dtype)
#
#print(agent_visibility_masks.shape)
#print(agent_data.shape)
#
#resets[0][0] = 1
#resets[0][1] = 2
#resets[0][2] = 2
#
#while True:
#    print("Stepping")
#    sim.step()
#    torchvision.utils.save_image((rgb_observations[0][0].float() / 255).permute(2, 0, 1), sys.argv[1])
#
#    print(prep_counter)
#    print(dones)
#    print(rewards[0][:4] * agent_valid_masks[0][:4])
#    print(agent_visibility_masks[0][:4] * agent_valid_masks[0][:4].unsqueeze(dim = 2))
#    print(box_visibility_masks[0][:4] * agent_valid_masks[0][:4].unsqueeze(dim = 2))
#    print(ramp_visibility_masks[0][:4] * agent_valid_masks[0][:4].unsqueeze(dim = 2))
#
#    action = get_keyboard_action()
#    
#    if action.reset < 0:
#        break
#
#    resets[0][0] = action.reset
#    actions[0][0][0] = action.x
#    actions[0][0][1] = action.y
#    actions[0][0][2] = action.r
#    actions[0][0][3] = action.g
#    actions[0][0][4] = action.l

del sim
