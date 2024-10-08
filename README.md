GPUDrive
========

![Python version](https://img.shields.io/badge/Python-3.11-blue) [![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/) [![Paper](https://img.shields.io/badge/arXiv-2408.01584-b31b1b.svg)](https://arxiv.org/abs/2408.01584)

GPUDrive is a GPU-accelerated, multi-agent driving simulator that runs at 1 million FPS. The simulator is written in C++, built on top of the [Madrona Game Engine](https://madrona-engine.github.io). We provide Python bindings and `gymnasium` wrappers in `torch` and `jax`, allowing you to interface with the simulator in Python using your preferred framework.

For more details, see our [paper](https://arxiv.org/abs/2408.01584) 📜 and the 👉 [introduction tutorials](https://github.com/Emerge-Lab/gpudrive/tree/main/examples/tutorials), which guide you through the basic usage.

<figure>
<img src="assets/GPUDrive_eval_with_humans_control_6.gif" alt="...">
<center><figcaption>Agents in GPUDrive can be controlled by any user-specified actor.</figcaption></center>
</figure>

## Implemented algorithms 🌱

| Algorithm      | Reference                                                                                                                           | README                                                                           |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **IPPO** | [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf) | [Source](https://github.com/Emerge-Lab/gpudrive/blob/main/baselines/ippo/README.md) |

## Installation 🛠️

To build GPUDrive, ensure you have all the dependencies listed [here](https://github.com/shacklettbp/madrona#dependencies). Briefly, you'll need a recent version of Python and CMake (>= version 3.22), as well as Xcode on macOS or Visual Studio on Windows.

Once you have the required dependencies, clone the repository (don't forget --recursive!):

```bash
git clone --recursive https://github.com/Emerge-Lab/gpudrive.git
cd gpudrive
```

---

<details>
  <summary>Optional: If you want to use the Madrona viewer in C++ (Not needed to render with pygame)</summary>

#### Extra dependencies to use Madrona viewer

  To build the simulator with visualization support on Linux (`build/viewer`), you will need to install X11 and OpenGL development libraries. Equivalent dependencies are already installed by Xcode on macOS. For example, on Ubuntu:

```bash
  sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev mesa-common-dev libc++1
```

</details>

---

Then, you can *choose* between two options for building the simulator:

---

<details>
  <summary>Option 1️⃣ : Manual install</summary>

For Linux and macOS, use the following commands:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j # cores to build with, e.g. 32
cd ..
```

For Windows, open the cloned repository in Visual Studio and build the project using the integrated `cmake` functionality.

Next, set up the Python components of the repository with pip:

```bash
pip install -e . # Add -Cpackages.madrona_escape_room.ext-out-dir=PATH_TO_YOUR_BUILD_DIR on Windows
```

</details>

---

---

<details>
  <summary>Option 2️⃣ : Poetry install</summary>

First create a conda environment using `environment.yml`:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate gpudrive
```

Run:

```bash
poetry install
```

</details>

---

Test whether the installation was successful by importing the simulator:

```Python
import gpudrive
```

## Getting started 🚀

To get started, see our [intro tutorials](https://github.com/Emerge-Lab/gpudrive/tree/main/examples/tutorials). These tutorials take approximately 30-60 minutes to complete and will guide you through the dataset, simulator, and how to populate the simulator with different types of actors.

<p align="center">
  <img src="assets/GPUDrive_docs_flow.png" width="1300" title="Getting started">
</p>

## Tests 📈

To further test the setup, you can run the pytests in the root directory:

```bash
pytest
```

To test if the simulator compiled correctly (and python lib did not), try running the headless program from the build directory.

```bash
cd build
./headless CPU 1 # Run on CPU, 1 step
```

## Pre-trained policy 🏋🏼‍♀️

We are open-sourcing a policy trained on 1,000 randomly sampled scenarios. After cloning the repository, Git LFS will automatically download the policy to the `models/learned_sb3_policy` directory. If you don’t find the file there, you can manually download the LFS files by running `git lfs pull`.

## Dataset `{ 🚦 🚗  🚙  🛣️ }`

### How to download the Waymo Open Motion Dataset

Two versions of the dataset are available:

- a mini-one that is about 1 GB and consists of 1000 training files and 100 validation / test files at: [Dropbox Link](https://www.dropbox.com/sh/8mxue9rdoizen3h/AADGRrHYBb86pZvDnHplDGvXa?dl=0).
- the full dataset (150 GB) and consists of 134453 training files and 12205 validation / test files: [Dropbox Link](https://www.dropbox.com/sh/wv75pjd8phxizj3/AABfNPWfjQdoTWvdVxsAjUL_a?dl=0)

The simulator supports initializing scenes from the `Nocturne` dataset. The input parameter for the simulator `json_path` takes in a path to a directory containing the files in the Nocturne format. The `SceneConfig` dataclass in `pygpudrive/env/config.py` dataclass is used to configure how scenes are selected from a folder with traffic scenarios.

## Citations

If you use GPUDrive in your work, please cite us:

```
@misc{kazemkhani2024gpudrivedatadrivenmultiagentdriving,
      title={GPUDrive: Data-driven, multi-agent driving simulation at 1 million FPS},
      author={Saman Kazemkhani and Aarav Pandya and Daphne Cornelisse and Brennan Shacklett and Eugene Vinitsky},
      year={2024},
      eprint={2408.01584},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2408.01584},
}
```

## Contributing and learning benchmark 👷‍♀️

If you find a bug of are missing features, please feel free to [create an issue or start contributing](https://github.com/Emerge-Lab/gpudrive/blob/main/CONTRIBUTING.md)! That link also points to a **learning benchmark** complete with training logs and videos of agent behaviors via `wandb`.

## Timeline

[![GPUDrive](https://api.star-history.com/svg?repos=Emerge-Lab/gpudrive&type=Date)](https://star-history.com/#Emerge-Lab/gpudrive&Date)
