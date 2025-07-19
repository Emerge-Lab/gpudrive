# Project Structure

This document outlines the structure of the GPUDrive project, detailing the purpose of each directory and key files within them.

## Root Directory

-   `CMakeLists.txt`: The main CMake file for building the C++ core of the simulator.
-   `Dockerfile`: Defines the Docker container for setting up the development and execution environment.
-   `README.md`: The main entry point for understanding the project, including installation and usage instructions.
-   `setup.py`: The setup script for installing the Python package.
-   `src/`: Contains the C++ source code for the simulation core.
-   `gpudrive/`: Contains the Python source code for the simulation environment, agents, and utilities.
-   `examples/`: Contains tutorial notebooks and example scripts.
-   `tests/`: Contains tests for both the C++ and Python components.
-   `assets/`: Contains assets used in the simulation and documentation, such as object files and GIFs.
-   `data/`: Directory for storing raw and processed dataset files.
-   `external/`: Contains external libraries and dependencies.

## `src` Directory

This directory contains the C++ source code for the simulation core, built on the Madrona engine.

-   `sim.hpp` / `sim.cpp`: Defines the main simulation logic, including ECS (Entity Component System) registration, task graph setup, and simulation state management.
-   `mgr.hpp` / `mgr.cpp`: The Manager class that acts as an interface between the C++ core and the external (Python) environment. It handles initialization, asset loading, and data marshalling.
-   `level_gen.hpp` / `level_gen.cpp`: Functions for generating and resetting the simulation world (levels).
-   `dynamics.hpp`: Contains different vehicle dynamics models (e.g., kinematic, bicycle).
-   `MapReader.hpp` / `MapReader.cpp`: Handles parsing of map data from files.
-   `bindings.cpp`: Contains the Python bindings for the C++ code using nanobind.

## `gpudrive` Directory

This directory contains the Python source code, which provides a `gymnasium`-compatible environment for the simulator.

-   `env/`:
    -   `base_env.py`: An abstract base class for the simulation environment.
    -   `env_torch.py`: A PyTorch-specific implementation of the environment.
    -   `env_jax.py`: A JAX-specific implementation of the environment.
    -   `config.py`: Configuration classes for the environment.
-   `agents/`: Contains different types of agents that can be used in the simulation.
-   `datatypes/`: Defines the data structures used for observations, actions, and other simulation data.
-   `networks/`: Contains neural network architectures for policies.
-   `utils/`: Utility functions used across the Python codebase.
-   `visualize/`: Tools for visualizing the simulation state and agent observations.

## `examples` Directory

This directory provides tutorials and examples to help users get started with GPUDrive.

-   `tutorials/`: A series of Jupyter notebooks that guide users through the basic functionalities of the simulator, from loading scenarios to using pre-trained agents.
