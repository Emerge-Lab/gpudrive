### Run on HPC


- Step 1: Request a node
```
srun --nodes=1 --tasks-per-node=1 --cpus-per-task=1 --mem=40GB --gres=gpu:a100:1 --time=02:00:00 --pty /bin/bash
```

- Step 2: Launch Singularity image

```
singularity exec --nv --overlay /scratch/dc4971/gpudrive/hpc/overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash
```

- Step 3: Activate conda env

```
conda activate gpudrive
```
