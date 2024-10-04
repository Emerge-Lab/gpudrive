GPUDrive
========

![Python version](https://img.shields.io/badge/Python-3.10-blue) [![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/) [![Paper](https://img.shields.io/badge/arXiv-2408.01584-b31b1b.svg)](https://arxiv.org/abs/2408.01584)

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

## Dataset

### Download the dataset

We provide two readily available datasets using GitHub Large File Storage (LFS). To access them, first install Git LFS by running:

```bash
git lfs install
```

Next, clone the repository as usual. Git LFS will automatically download the dataset when you pull the repository.

The datasets will be available at the locations:

- Validation dataset (150 scenarios): `data/processed/validation`
- Test dataset (150 scenarios): `data/processed/testing`

### Re-building the dataset

GPUDrive is compatible with the complete [Waymo Open Motion Dataset](https://github.com/waymo-research/waymo-open-dataset), which contains over 100,000 scenarios. To download new files and create scenarios for the simulator, follow these three steps.

1. First, head to [https://waymo.com/open/](https://waymo.com/open/) and click on the "download" button a the top. After registering, click on the files from `v1.2.1 March 2024`, the newest version of the dataset at the time of wrting (10/2024). This will lead you a Google Cloud page. From here, you should see a folder structure like this:

```
waymo_open_dataset_motion_v_1_2_1/
│
├── uncompressed/
│   ├── lidar_and_camera/
│   ├── scenario/
│   │   ├── testing_interactive/
│   │   ├── testing/
│   │   ├── training_20s/
│   │   ├── training/
│   │   ├── validation_interactive/
│   │   └── validation/
│   └── tf_example/
```

2. Now, download files from testing, training and/or validation in the **`scenario`** folder. An easy way to do this is through `gsutil`.  First register using:

```bash
gcloud auth login
```

...then run the command below to download the dataset you prefer. For example, to download the validation dataset:

```bash
gsutil -m cp -r gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/scenario/validation/ data/raw
```

where `data/raw` is your local storage folder. Note that this can take a while, depending on the size of the dataset you're downloading.

3. The last thing we need to do is convert the raw data to a format that is compatible with the simulator using:

```bash
python data_utils/process_waymo_files.py '<raw-data-path>' '<storage-path>' '<dataset>'
```

For example, if you want to only process the validation data:

```bash
python data_utils/process_waymo_files.py 'data/raw/' 'data/processed/' 'validation'
>>>
Processing Waymo files: 100%|████████████████████████████████████████████████████████████████| 150/150 [00:05<00:00, 28.18it/s]
INFO:root:Done!
```

and that's it!

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
