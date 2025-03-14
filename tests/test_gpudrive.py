import pufferlib

from gpudrive.env.env_puffer import PufferGPUDrive

env = PufferGPUDrive()
env.reset()

for i in range(10):
    actions = env.action_space.sample()
    env.step(actions)

