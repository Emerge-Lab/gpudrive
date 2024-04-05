GPUDrive
============================

This is an Batch RL environment simulator of Nocturne built on the [Madrona Engine](https://madrona-engine.github.io). It supports loading multiple worlds with multi agent support. Python bindings are exposed for easy integration with RL algorithms.

The Environment and Learning Task
--------------

The codebase trains a shared policy that controls agents individually with direct engine inputs rather than pixel observations. Agents interact with the simulator as follows:

### Action Space:
 * Acceleration: Continuous float values for acceleration applied to the agents.
 * Steering angle: Continuous float values for steering angle applied according to the bicycle kinematic model.
 * Heading angle (currently unused): Continuous float values for heading angle. This controls where the agent is looking.

### Observation Space:

**SelfObservation**

The `SelfObservation` tensor of shape `(5,)` for each agent provides information about the agent's own state. The respective values are 

- `SelfObservation[0]`: Represents the current *speed* of the agent.
- `SelfObservation[1:3]` : *length* and *width* of the agent.
- `SelfObservation[3:5]`: *Coordinates (x,y)* of the goal relative to the agent. 

**PartnerObservation**

The `PartnerObservation` tensor of shape `(num_agents-1,7)` for each agent provides information about other agents in the range `params.observationRadius`. All the values in this tensor are *relative to the ego agent*. The respective values for each `PartnerObservation` are

- `PartnerObservation[0]`: The *speed* of the observed neighboring agent.
- `PartnerObservation[1:3]`: The *position (x,y)* of the observed neighbouring agent.
- `PartnerObservation[3]`: The *orientation* of the neighboring agent.
- `PartnerObservation[4:6]`: The *length* and *width* of the neighbouring agent.
- `PartnerObservation[6]`: The type of agent. 

**AgentMapObservations**

The `AgentMapObservations` tensor of shape (num_road_objs, 4) for each agent provides information about the road objects in the range `params.observationRadius`. All the values in this tensor are *relative to the ego agent*. The respective values for each `AgentMapObservations` are

- `AgentMapObservations[0:2]`: The position coordinates for the road object.
- `AgentMapObservations[2]`: The relative orientation of the road object.
- `AgentMapObservations[3]`: The road object type.

**Rewards:**
  Agents are rewarded for the max distance achieved along the Y axis (the length of the level). Each step, new reward is assigned if the agents have progressed further in the level, or a small penalty reward is assigned if not.
 
For specific details about the format of observations, refer to exported ECS components introduced in the [code walkthrough section](#simulator-code-walkthrough-learning-the-madrona-ecs-apis). 

Overall the "full simulator" contains logic for three major concerns:
* Procedurally generating a new random level for each episode.
* Time stepping the environment, which includes executing rigid body physics and evaluating game logic in response to agent actions.
* Generating agent observations from the state of the environment, which are communicated as PyTorch tensors to external policy evaluation or learning code.

Build Instructions
--------
First, make sure you have all the dependencies listed [here](https://github.com/shacklettbp/madrona#dependencies) (briefly, recent python and cmake, as well as Xcode or Visual Studio on MacOS or Windows respectively).

To build the simulator with visualization support on Linux (`build/viewer`), you also need to install X11 and OpenGL development libraries. Equivalent dependencies should already be installed by Xcode on MacOS.
For example, on Ubuntu:
```bash
sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev mesa-common-dev libc++1

```

The built-in training functionality requires [PyTorch 2.0](https://pytorch.org/get-started/locally/) or later as well.

Now that you have the required dependencies, fetch the repo (don't forget `--recursive`!):
```bash
git clone --recursive https://github.com/Emerge-Lab/gpudrive.git
cd gpudrive
```

# Manual Install

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
pip install -e . # Add -Cpackages.madrona_escape_room.ext-out-dir=PATH_TO_YOUR_BUILD_DIR on Windows
```

You can then view the environment by running:
```bash
./build/viewer
```

Or test the PyTorch training integration:
```bash
python scripts/train.py --num-worlds 1024 --num-updates 100 --ckpt-dir build/ckpts
```

# Poetry install

### Conda 

Create a conda environment using `environment.yml` and then run `poetry install`

```bash
conda env create -f environment.yml`
poetry install
```



Simulator Code Walkthrough (Learning the Madrona ECS APIs)
-----------------------------------------------------------

As mentioned above, this repo is intended to serve as a tutorial for how to use Madrona to implement a batch simulator for a simple 3D environment. If you're not interested in implementing your own novel environment simulator in Madrona and just want to try training agents, [skip to the next section](#training-agents).

We assume the reader is familiar with the key concepts of the entity component system (ECS) design pattern.  If you are unfamiliar with ECS concepts, we recommend that you check out Sander Mertens' very useful [Entity Components FAQ](https://github.com/SanderMertens/ecs-faq). 

#### Defining the Simulation's State: Components and Archetypes ####

The first step to understanding the simulator's implementation is to understand the ECS components that make up the data in the simulation. All the custom logic in the simulation (as well as logic for built-in systems like physics) is written in terms of these data types. Take a look at [`src/types.hpp`](https://github.com/shacklettbp/madrona_escape_room/blob/main/src/types.hpp#L28). This file first defines all the ECS components as simple C++ structs and next declares the ECS archetypes in terms of the components they are composed of. For integration with learning, many of the components of the `Agent` archetype are directly exported as PyTorch tensors. For example, the `Action` component directly correspondes to the action space described above, and `RoomEntityObservations` is the agent observations of all the objects in each room.

#### Defining the Simulation's Logic: Systems and the Task Graph ####

After understanding the ECS components that make up the data of the simulation, the next step is to learn about the ECS systems that operate on these components and implement the custom logic of the simulation. Madrona simulators define a centralized task graph that declares all the systems that need to execute during each simulation step that the Madrona runtime then executes across all the unique worlds in a simulation batch simultaneously for each step. This codebase builds the task graph during initialization in the [`Sim::setupTasks`](https://github.com/shacklettbp/madrona_escape_room/blob/main/src/sim.cpp#L552) function using `TaskGraphBuilder` class provided by Madrona. Take note of all the ECS system functions that `setupTasks` enqueues in the task graph using `ParallelForNode<>` nodes, and match the component types to the components declared you viewed in [`types.hpp`](https://github.com/shacklettbp/madrona_escape_room/blob/main/src/types.hpp). For example, `movementSystem`, added at the beginning of the task graph, implements the custom logic that translates discrete agent actions from the `Action` component into forces for the physics engine. At the end of each step, `collectObservationSystem` reads the simulation state and builds observations for the agent policy.

At this point for an overview of the whole simulator you can continue to the next section, or for further details, you can continue reading [`src/sim.cpp`](https://github.com/shacklettbp/madrona_escape_room/blob/main/src/sim.cpp) and ['src/sim.hpp](https://github.com/shacklettbp/madrona_escape_room/blob/main/src/sim.hpp) where all the core simulation logic is located with the exception of level generation logic that handles creating new entities and placing them. The level generation logic starts with the [`generateWorld`](https://github.com/shacklettbp/madrona_escape_room/blob/main/src/level_gen.cpp#L558) function in [`src/level_gen.cpp`](https://github.com/shacklettbp/madrona_escape_room/blob/main/src/level_gen.cpp) and is called for each world when a training episode ends.

#### Initializing the Simulator and Interfacing with Python Training Code ####

The final missing pieces of the simulator are how the Madrona backends are initialized and how data communication between PyTorch and the simulator is managed. These pieces are controlled by the `Manager` class in [`src/mgr.hpp`](https://github.com/shacklettbp/madrona_escape_room/blob/main/src/mgr.hpp) and [`src/mgr.cpp`](https://github.com/shacklettbp/madrona_escape_room/blob/main/src/mgr.cpp). During initialization, the `Manager` constructor is passed an `ExecMode` object from pytorch that dictates whether the CPU or CUDA backends should be initialized. The `Manager` class then loads physics assets off disk (copying them to the GPU if needed) and then initializes the appropriate backend. Once initialization is complete, the python code can access simulation state through the `Manager`'s exported PyTorch tensors (for example, `Manager::rewardTensor`) via the python bindings declared in [`src/bindings.cpp`](https://github.com/shacklettbp/madrona_escape_room/blob/main/src/mgr.cpp). These bindings are just a thin wrapper around the `Manager` class using [`nanobind`](https://github.com/wjakob/nanobind).

#### Visualizing Simulation Output ####

The code that integrates with our visualization infrastructure is located in [`src/viewer.cpp`](https://github.com/shacklettbp/madrona_escape_room/blob/main/src/viewer.cpp). This code links with the `Manager` class and produces the `viewer` binary in the build directory that lets you control the agents directly and replay actions. More customization in the viewer code to support custom UI and overlays will be supported in the future.

Training Agents 
--------------------------------

In addition to the simulator itself, this repo contains a simple PPO implementation (in PyTorch) to demonstrate how to integrate a training codebase with a Madrona batch simulator. [`scripts/train.py`](https://github.com/shacklettbp/madrona_escape_room/blob/main/scripts/train.py) is the training code entry point, while the bulk of the PPO implementation is in [train_src/madrona_escape_room_learn](https://github.com/shacklettbp/madrona_escape_room/blob/main/train_src/madrona_escape_room_learn). 

For example, the following settings will produce agents that should be able to solve all three rooms fairly consistently:
```bash
python scripts/train.py --num-worlds 8192 --num-updates 5000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints/
```

If your machine doesn't support the GPU backend, simply remove the `--gpu-sim` argument above and consider reducing the `--num-worlds` argument to reduce the batch size. 

After 5000 updates, the policy should have finished training. You can run the policy and record a set of actions with:
```bash
python scripts/infer.py --num-worlds 1 --num-steps 1000 --fp16 --ckpt-path build/checkpoints/5000.pth --action-dump-path build/dumped_actions
```

Finally, you can replay these actions in the `viewer` program to see how your agents behave:
```bash
./build/viewer 1 --cpu build/dumped_actions
```

Hold down right click and use WASD to fly around the environment, or use controls in the UI to following a viewer in first-person mode. Hopefully your agents perform similarly to those in the video at the start of this README!

Note that the hyperparameters chosen in scripts/train.py are likely non-optimal. Let us know if you find ones that train faster.

Citation
--------
If you use Madrona in a research project, please cite our SIGGRAPH paper.

```
@article{...,
}
```
