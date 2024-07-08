GPUDrive
============================

GPUDrive is a GPU-accelerated, multi-agent driving simulator that runs at 1 million FPS. The simulator is written in C++, built on top of the [Madrona Game Engine](https://madrona-engine.github.io). We provide Python bindings and `gymnasium` wrappers in `torch` and `jax`, allowing you to interface with the simulator in Python using your preferred framework.

For more details, see our [paper]() and the [introduction tutorials](https://github.com/Emerge-Lab/gpudrive/tree/main/examples/tutorials), which guide you through the basic usage.

<figure>
<img src="docs/assets/gpudrive_gif_collage.gif" alt="...">
<center><figcaption>Three example scenarios from a bird's eye view.</figcaption></center>
</figure>

## Implemented algorithms 🌱

| Algorithm | Reference | README |
|----------|----------|----------|
|   **IPPO**   |   [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf)   |   [Source](https://github.com/Emerge-Lab/gpudrive/blob/main/baselines/ippo/README.md)   |


## Installation 🛠️

To build GPUDrive, ensure you have all the dependencies listed [here](https://github.com/shacklettbp/madrona#dependencies). Briefly, you'll need a recent version of Python and CMake, as well as Xcode on macOS or Visual Studio on Windows.

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

- **Option 2**:  Poetry install

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
```Python
import gpudrive
```

## Getting started 🚀

To get started, see our [intro tutorials](https://github.com/Emerge-Lab/gpudrive/tree/main/examples/tutorials). These tutorials take approximately 30-60 minutes to complete and will guide you through the dataset, simulator, and gymnasium wrappers.

<p align="center">
  <img src="docs/assets/navigation.png" width="350" title="Getting started">
</p>


## Dataset `{ 🚦 🚗  🚙  🛣️ }`

### How to download the Waymo Open Motion Dataset
Two versions of the dataset are available:
- a mini-one that is about 1 GB and consists of 1000 training files and 100 validation / test files at: [Dropbox Link](https://www.dropbox.com/sh/8mxue9rdoizen3h/AADGRrHYBb86pZvDnHplDGvXa?dl=0).
- the full dataset (150 GB) and consists of 134453 training files and 12205 validation / test files: [Dropbox Link](https://www.dropbox.com/sh/wv75pjd8phxizj3/AABfNPWfjQdoTWvdVxsAjUL_a?dl=0)

The simulator supports initializing scenes from the Nocturne dataset. The input parameter for the simulator `json_path` takes in a path to a directory containing the files in the Nocturne format. The directory should contain a `valid_files.json` with a list of the files to be initialized.


## Tests 📈

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

How to sample the set of scenarios you want to train on can be set using `sample_method`.

| `sample_method` | Description |
|----------|-------------|
| **first_n** | Takes the first `num_worlds` files. Fails if the number of files is less than `num_worlds`. |
| **random_n** | Randomly selects `num_worlds` files from the dataset. Fails if the number of files is less than `num_worlds`. |
| **pad_n** | Initializes as many files as available up to `num_worlds`, then repeats the first file to pad until `num_worlds` files are loaded. Fails if there are more files than `num_worlds`. |
| **exact_n** | Initializes exactly `num_worlds` files, ensuring that the count matches precisely with no more or less. |


## Rendering

Render settings can be changed using the `RenderConfig`.

| `Render Mode` | Description
|--|--|
| **PYGAME_ABSOLUTE** | Renders the absolute view of the scene with all the agents. Returns a single frame for a world.
| **PYGAME_EGOCENTRIC** | Renders the egocentric view for each agent in a scene. Returns `num_agents` frames for each world.
| **PYGAME_LIDAR** | Renders the Lidar views for an egent in a scene if Lidar is enabled. Returns `num_agents` frames for each world.

Resolution of the frames can be specified using the `resolution` param which takes in a tuple of (W,H).

Below are the renders for each mode
<table>
  <tr>
    <td>
      <figure>
        <img src="../../data/absolute.gif" alt="Absolute">
        <center><figcaption>Absolute</figcaption></center>
      </figure>
    </td>
    <td>
      <figure>
        <img src="../../data/Egocentric.gif" alt="Egocentric">
        <center><figcaption>Egocentric</figcaption></center>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="../../data/Lidar360.gif" alt="Lidar with 360 FOV">
        <center><figcaption>Lidar with 360 FOV</figcaption></center>
      </figure>
    </td>
    <td>
      <figure>
        <img src="../../data/Lidar120.gif" alt="Lidar with 120 FOV">
        <center><figcaption>Lidar with 120 FOV</figcaption></center>
      </figure>
    </td>
  </tr>
</table>

## Sharp Bits

TODO(dc)

## Citations

If you use GPUDrive in your work, please cite us:
TODO(dc)


The Waymo Open Dataset is discussed in the following publication:

```
@misc{ettinger2021large,
      title={Large Scale Interactive Motion Forecasting for Autonomous Driving : The Waymo Open Motion Dataset},
      author={Scott Ettinger and others},
      year={2021},
      eprint={2104.10133},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
