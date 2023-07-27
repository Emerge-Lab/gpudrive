Madrona 3D Example Simulator
============================

This is an example RL environment simulator built on the [Madrona Engine](https://madrona-engine.github.io). 
The goal of this repository is to demonstrate how to use Madrona's ECS APIs and 
how to interface with the engine's rigid body physics and rendering functionality.
Additionally, to demonstrate how to integrate the simulator with python training code, we show how to connect the simulator up to a PyTorch PPO training loop. 
We provide a simple training script that performs end-to-end agent training using PPO.

If you're interested in using Madrona to implement a high-performance batch simulator for a new environment or RL training task, we highly recommend forking this repo and adding/removing code as needed, rather than starting from scratch. This will ensure the build system and backends are setup correctly.

The Environment and Learning Task
--------------

As shown below, the simulator randomly creates a set of 3 rooms where agents need to step on buttons and pull blocks to open a door and advance to the next room. Agents are rewarded based on their total progress along the length of the level.

[SMALL VIDEO CLIP HERE]

Build Instructions
--------
First, make sure you have all the dependencies listed [here](https://github.com/shacklettbp/madrona#dependencies) (briefly, recent python and cmake, as well as Xcode or Visual Studio on MacOS or Windows respectively).

Additionally, the rendering functionality in this repository requires the X11 development headers on Linux (e.g. the `libx11-dev` package on Ubuntu). The training functionality requires PyTorch 2.0 or later as well.

Next, fetch the repo (don't forget `--recursive`!):
```bash
git clone --recursive https://github.com/shacklettbp/madrona_3d_example.git
cd madrona_3d_example
```

Next, for Linux and MacOS: Run `cmake` and then `make` to build the simulator:
```bash
mkdir build
cd build
cmake ..
make -j # cores to build with
cd ..
```

Or on Windows, open the cloned repository in Visual Studio and build
the project using the integrated `cmake` functionality.


Now, setup the python components of the repository with `pip`:
```bash
pip install -e . # Add -Cpackages.madrona_3d_example.ext-out-dir=PATH_TO_YOUR_BUILD_DIR on Windows
```

You can then view the environment by running:
```bash
./build/viewer
```

Or test the PyTorch training integration: (first, [install pytorch](https://pytorch.org/get-started/locally/))
```bash
python scripts/train.py --num-worlds 1024 --num-updates 100
```

Simulator Code Walkthrough (Learning the Madrona ECS APIs)
-----------------------------------------------------------

As mentioned above, this repo is intended to serve as a tutorial for how to use Madrona to implement a batch simulator for a simple 3D environment. If you're not interested in implementing your own novel environment simulator in Madrona and just want to try training agents, skip to the next section.

We assume the reader is familiar with the key concepts of the entity component system (ECS) design pattern.  If you are unfamiliar with ECS concepts, we recommend that you check out Sander Martens' very useful [Entity Components FAQ](https://github.com/SanderMertens/ecs-faq). 

#### Defining Simulation State: Components and Archetypes ####

Take a look at [`src/types.hpp`](https://github.com/shacklettbp/madrona_3d_example/blob/main/src/types.hpp#L28). This file defines all the simulator's custom ECS components and archetypes. In particular, the `Agent` archetype defines all the components used by the agents in the simulation. Many of the `Agent` components are directly exported as PyTorch tensors.

#### Defining Simulator Logic: Systems and the Task Graph ####

To get an understanding of the simulation loop, view [`Sim::setupTasks`](https://github.com/shacklettbp/madrona_3d_example/blob/main/src/sim.cpp#L552). This function builds the task graph that defines the logic for each simulation step. Keep in mind this logic is carried out for all the unique worlds in an simulation batch. Take note of the ECS system functions (`movementSystem`, `collectObservationsSystem`, etc) that `setupTasks` enqueues into the task graph and the components they iterate over.

At this point, you can continue reading [`src/sim.cpp`](https://github.com/shacklettbp/madrona_3d_example/blob/main/src/sim.cpp) and ['src/sim.hpp](https://github.com/shacklettbp/madrona_3d_example/blob/main/src/sim.hpp) where all the core simulation logic is located, or visit the [`generateWorld`](https://github.com/shacklettbp/madrona_3d_example/blob/main/src/level_gen.cpp#L558) function in [`src/level_gen.cpp`](https://github.com/shacklettbp/madrona_3d_example/blob/main/src/level_gen.cpp) to see how the levels are randomly generated.

#### Connecting the Simulator to Python Training Code ####

To see the bridging code that manages the data communication between PyTorch training code and the batch simulator, check out the `Manager` class in [`src/mgr.hpp`](https://github.com/shacklettbp/madrona_3d_example/blob/main/src/mgr.hpp) and [`src/mgr.cpp`](https://github.com/shacklettbp/madrona_3d_example/blob/main/src/mgr.cpp). On a related note, the python binding code lives in [`src/bindings.cpp`](https://github.com/shacklettbp/madrona_3d_example/blob/main/src/mgr.cpp). These bindings are just a thin wrapper around the `Manager` class using [`nanobind`](https://github.com/wjakob/nanobind).

#### Visualizing Simulation Output ####

The code that integrates with our visualization infrastructure is located in [`src/viewer.cpp`](https://github.com/shacklettbp/madrona_3d_example/blob/main/src/viewer.cpp). This code links with the `Manager` class and produces the `viewer` binary in the build directory that lets you control the agents directly and replay actions. More customization in the viewer code to support custom UI and overlays will be supported in the future.

Training Agents 
--------------------------------

In addition to the simulator itself, this repo contains a simple PPO implementation (in PyTorch) to demonstrate how to integrate a training codebase with a Madrona batch simulator. [`scripts/train.py`](https://github.com/shacklettbp/madrona_3d_example/blob/main/scripts/train.py) is the training code entry point.

For example, the following settings will produce agents that can at least solve the first two rooms in just a few minutes on a RTX 3090:
```bash
python scripts/train.py --num-worlds 8192 --num-updates 5000 --gpu-sim --ckpt-dir build/checkpoints/
```

If your machine doesn't support the GPU backend, simply remove the `--gpu-sim` argument above and consider reducing the `--num-worlds` argument to reduce the batch size. 

After 5000 updates, the policy should have finished training. You can run the policy and record a set of actions with:
```bash
python scripts/infer.py --num-worlds 1 --num-steps 1000 --ckpt-path build/checkpoints/5000.pth --action-dump-path build/dumped_actions
```

Finally, you can replay these actions in the `viewer` program to see how your agents behave:
```bash
./build/viewer --cpu 1 build/dumped_actions
```

Hold down right click and use WASD to fly around the environment, or use controls in the UI to following a viewer in first-person mode. Hopefully your agents perform similarly to those in the video at the start of this README!

Citation
--------
If you use Madrona in a research project, please cite our SIGGRAPH paper!

```
@article{shacklett23madrona,
    title   = {An Extensible, Data-Oriented Architecture for High-Performance, Many-World Simulation},
    author  = {Brennan Shacklett and Luc Guy Rosenzweig and Zhiqiang Xie and Bidipta Sarkar and Andrew Szot and Erik Wijmans and Vladlen Koltun and Dhruv Batra and Kayvon Fatahalian},
    journal = {Transactions on Graphics},
    year    = {2023}
}
```
