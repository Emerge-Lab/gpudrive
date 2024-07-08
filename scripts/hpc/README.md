## Setup of `gpudrive` on HPC

### Installation (only do the first time)

1. Clone repo into HPC Greene at location `/scratch/$USER`, where `$USER` is your netid.
2. Move into `gpudrive` folder by running: `cd gpudrive`
3. Set up Singularity image: `bash ./hpc/launch.sh`
4. Create a conda environment: `conda env create -f environment.yml`
5. Activate venv: `conda activate gpudrive`
6. Install `gpudrive`: `poetry install`
7. Restart Singularity image: `exit`
8. Relaunch image: `bash ./hpc/launch.sh`
9. Check if `gpudrive` is properly installed:
    - (a) launch a Python shell
    - (b) import `gpudrive` by running `import gpudrive`

### Usage

Request an interactive compute node:
```shell
# Example: Request a single GPU for one hour
srun --nodes=1 --tasks-per-node=1 --cpus-per-task=1 --mem=40GB --gres=gpu:1 --time=1:00:00 --pty /bin/bash
```

Navigate to the repo:
```shell
cd /scratch/$USER/gpudrive
```

Launch the Singularity image:
```shell
bash ./hpc/launch.sh
```

Activate the conda environment:
```shell
conda activate gpudrive
```
