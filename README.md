GPUDrive: Data-driven, multi-agent driving simulation at 1 million FPS
============================

GPUDrive is a batched simulator of Nocturne, built on the [Madrona Engine](https://madrona-engine.github.io). It supports multiple worlds and multi-agent environments. Python bindings are available for seamless integration with RL algorithms.

For more details, see our [paper]() and the [introduction tutorials](https://github.com/Emerge-Lab/gpudrive/tree/main/examples/tutorials), which guide you through the basic usage.



<figure>
<img src="data/gpudrive_gif_collage.gif" alt="...">
<center><figcaption>Three example scenarios from a bird's eye view.</figcaption></center>
</figure>

## Baseline algorithms


| Algorithm | Reference | README |
|----------|----------|----------|
|   **IPPO**   |   [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf)   |   [Source](https://github.com/Emerge-Lab/gpudrive/blob/main/baselines/ippo/README.md)   |


## Installation

To build GPUDrive, ensure you have all the dependencies listed [here](https://github.com/shacklettbp/madrona#dependencies). Briefly, you'll need a recent version of Python and CMake, as well as Xcode on macOS or Visual Studio on Windows.

To build the simulator with visualization support on Linux (`build/viewer`), you also need to install X11 and OpenGL development libraries. Equivalent dependencies are already installed by Xcode on macOS. For example, on Ubuntu:
```bash
sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev mesa-common-dev libc++1
```

Once you have the required dependencies, clone the repository (don't forget --recursive!):
```bash
git clone --recursive https://github.com/Emerge-Lab/gpudrive.git
cd gpudrive
```

Then, you can choose between two options for building the simulator:

- **Option 1**: Manual Install

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

- **Option 2**:  Poetry Install

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

Test whether the installation was successful by importing the simulator:
```bash
import gpudrive
```

## Getting started 🚀

To get started, see our [intro tutorials](https://github.com/Emerge-Lab/gpudrive/tree/main/examples/tutorials). These tutorials take approximately 30-60 minutes to complete and will guide you through the dataset, simulator, and gymnasium wrappers.

<p align="center">
  <img src="data/navigation.png" width="350" title="Getting started">
</p>


## Tests


To further test the setup, you can run the pytests in the root directory:
```bash
pytest
```

To test if the simulator compiled correctly (and python lib did not), try running the headless program from the build directory. Remember to change the location of the data in `src/headless.cpp` and compiling again before running it.

```bash
cd build
./headless CPU 1 1 # Run on CPU , 1 world, 1 step
```


## Dataset

### How to download the Waymo Open Motion Dataset
Two versions of the dataset are available:
- a mini-one that is about 1 GB and consists of 1000 training files and 100 validation / test files at: [Dropbox Link](https://www.dropbox.com/sh/8mxue9rdoizen3h/AADGRrHYBb86pZvDnHplDGvXa?dl=0).
- the full dataset (150 GB) and consists of 134453 training files and 12205 validation / test files: [Dropbox Link](https://www.dropbox.com/sh/wv75pjd8phxizj3/AABfNPWfjQdoTWvdVxsAjUL_a?dl=0)

The simulator supports initializing scenes from the Nocturne dataset. The input parameter for the simulator `json_path` takes in a path to a directory containing the files in the Nocturne format. The directory should contain a `valid_files.json` with a list of the files to be initialized.


Citation
--------
This batch-simulator is made in thanks to Madrona Engine.

```
@article{...,
}
```
