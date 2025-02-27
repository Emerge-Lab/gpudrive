"""
A script for generating sbatch array submission scripts.
Based on https://github.com/TysonRayJones/PythonTools/tree/master
"""
import numpy as np
import os
import re
from typing import Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants (from .env)
LOG_FOLDER = os.getenv("LOG_FOLDER")
HPC_ACCOUNT = os.getenv("NYU_HPC_ACCOUNT")
USERNAME = os.getenv("USERNAME")
SINGULARITY_IMAGE = os.getenv("SINGULARITY_IMAGE")
OVERLAY_FILE = os.getenv("OVERLAY_FILE")

# Set to python file to run
RUN_FILE = "baselines/ppo/ppo_pufferlib.py"

# Default SLURM fields
DEFAULT_SLURM_FIELDS = {
    "num_nodes": 1,
    "num_cpus": 1,
    "num_gpus": 1,
    "gpu_type": None,  # --gres=gpu:1:rtx8000; logic: if gpu_type in supported list, add to end. If not supported list, throw exception, and if not provided, don't add GPU type
    "memory": 10,
    "memory_unit": "GB",
    "time_d": 0,
    "time_h": 0,
    "time_m": 0,
    "time_s": 0,
    "max_sim_jobs": None,
    "output": f"{LOG_FOLDER}output_%A_%a.txt",
    "error": f"{LOG_FOLDER}error_%A_%a.txt",
    "account": HPC_ACCOUNT,
    "username": USERNAME,
    "singularity_image": SINGULARITY_IMAGE,
    "overlay_file": OVERLAY_FILE,
    "run_file": RUN_FILE,
}

# a template for the submit script
# (bash braces must be escaped by doubling: $var = ${{var}})
# num_jobs, param_arr_init, param_val_assign and param_list are special fields

TEMPLATE_SBATCH = """
#!/bin/bash

#SBATCH --array=0-{num_jobs}%{max_sim_jobs}
#SBATCH --job-name={job_name}
#SBATCH --output={output}
#SBATCH --error={error}
#SBATCH --mem={memory}{memory_unit}
#SBATCH --time={time_d}-{time_h}:{time_m}:{time_s}
#SBATCH --nodes={num_nodes}
#SBATCH --cpus-per-task={num_cpus}
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --account={account}

SINGULARITY_IMAGE={singularity_image}
OVERLAY_FILE=/scratch/{username}/{overlay_file}

singularity exec --nv --overlay "${{OVERLAY_FILE}}:ro" \
    "${{SINGULARITY_IMAGE}}" \
    /bin/bash

echo "Successfully launched image."

{param_arr_init}

trial=${{SLURM_ARRAY_TASK_ID}}
{param_val_assign}

# Source the Conda setup script to make the `conda` command available
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh

# Activate conda environment
conda activate /scratch/{username}/.conda/gpudrive

# Set up SSL certificates for wandb logging
export SSL_CERT_FILE=$(python -m certifi)
export REQUESTS_CA_BUNDLE=$(python -m certifi)

# Run PPO
python {run_file} {param_cli_list}
""".strip()


def _mth(exp):
    return "$(( %s ))" % exp


def _len(arr):
    return "${{#%s[@]}}" % arr


def _get(arr, elem):
    return "${{%s[%s]}}" % (arr, elem)


def _eq(var, val):
    return "%s=%s" % (var, val)


def _op(a, op, b):
    return _mth("%s %s %s" % (a, op, b))


def _arr(arr):
    return "( %s )" % " ".join(map(str, arr))


def _seq(a, b, step):
    return "($( seq %d %d %d ))" % (a, step, b)


def _var(var):
    return "${%s}" % var


def _cli_var(var):
    tmp = f"--{var}".replace("_", "-")
    return f"{tmp}=${{{var}}}"


# Templates for param array construction and element access
PARAM_ARR = "{param}_values"
PARAM_EXPRS = {
    "param_arr_init": _eq(PARAM_ARR, "{values}"),
    "param_val_assign": {
        "assign": _eq(
            "{param}", _get(PARAM_ARR, _op("trial", "%", _len(PARAM_ARR)))
        ),
        "increment": _eq("trial", _op("trial", "/", _len(PARAM_ARR))),
    },
}


def _to_bash(obj):
    if isinstance(obj, range):
        return _seq(obj.start, obj.stop - 1, obj.step)
    if isinstance(obj, list) or isinstance(obj, tuple):
        return _arr(obj)
    raise ValueError("Unknown object type %s" % type(obj).__name__)


def _get_params_bash(params, values):
    # Get lines of bash code for creating/accessing param arrays
    init_lines = []
    assign_lines = []
    init_temp = PARAM_EXPRS["param_arr_init"]
    assign_temps = PARAM_EXPRS["param_val_assign"]

    for param, vals in zip(params, values):
        init_lines.append(init_temp.format(param=param, values=_to_bash(vals)))
        assign_lines.append(assign_temps["assign"].format(param=param))
        assign_lines.append(assign_temps["increment"].format(param=param))

    # Remove superfluous final trial reassign
    assign_lines.pop()

    return init_lines, assign_lines


def get_script(
    fields: Dict = DEFAULT_SLURM_FIELDS, params: Dict = {}, param_order=None
):
    """
    returns a string of a SLURM submission script using the passed fields
    and which creates an array of jobs which sweep the given params

    fields:      dict of SLURM field names to their values. type is ignored
    params:      a dict of (param names, param value list) pairs.
                 The param name is the name of the bash variable created in
                 the submission script which will contain the param's current
                 value (for that SLURM job instance). param value list is
                 a list (or range instance) of the values the param should take,
                 to be run once against every other possible configuration of all params.
    param_order: a list containing all param names which indicates the ordering
                 of the params in the sweep. The last param changes every
                 job number. If not supplied, uses an arbitrary order
    """

    assert isinstance(fields, dict)
    assert isinstance(params, dict)
    assert (
        isinstance(param_order, list)
        or isinstance(param_order, tuple)
        or param_order == None
    )
    if param_order == None:
        param_order = list(params.keys())

    # Check each field appears in the template
    for field in fields:
        if ("{%s}" % field) not in TEMPLATE_SBATCH:
            raise ValueError("passed field %s unused in template" % field)

    # Calculate total number of jobs (minus 1; SLURM is inclusive)
    num_jobs = 1
    for vals in params.values():
        num_jobs *= len(vals)
    num_jobs -= 1

    # Get bash code for param sweeping
    init_lines, assign_lines = _get_params_bash(
        param_order, [params[key] for key in param_order]
    )

    # Build template substitutions (overriding defaults)
    subs = {
        "param_arr_init": "\n".join(init_lines),
        "param_val_assign": "\n".join(assign_lines),
        "param_cli_list": " ".join(map(_cli_var, param_order)),
        "num_jobs": num_jobs,
    }

    for key, val in DEFAULT_SLURM_FIELDS.items():
        subs[key] = val
    for key, val in fields.items():
        subs[key] = val
    if "job_name" not in subs:
        subs["job_name"] = "my_job"

    return TEMPLATE_SBATCH.format(**subs)


def save_script(filename, file_path, fields, params, param_order=None):
    """Generate and save sbatch (.sh) submission script."""

    sbatch_script = get_script(fields, params, param_order)

    if not file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path + filename, "w") as file:
        file.write(sbatch_script)


if __name__ == "__main__":

    group = "02_24_S10_000"

    fields = {
        "time_h": 47,  # Max time per job (job will finish if run is done before)
        "num_gpus": 1,  # GPUs per job
        "max_sim_jobs": 30,  # Max jobs at the same time
        "memory": 70,
        "job_name": group,
    }
    
    hyperparams = {
        "group": [group], # Group name
        "num_worlds": [800],
        "resample_scenes": [1], # Yes
        "k_unique_scenes": [800],
        "resample_interval": [5_000_000],
        "total_timesteps": [4_000_000_000],
        "resample_dataset_size": [10_000],
        "batch_size": [524288],
        "minibatch_size": [16384],
        "update_epochs": [4],
        "ent_coef": [0.001, 0.003, 0.0001],
        "render": [0],
        #"seed": [42, 3],
    }

    save_script(
        file_path="examples/experimental/sbatch_scripts/",
        filename=f"sbatch_{group}.sh",
        fields=fields,
        params=hyperparams,
    )

    # hyperparams = {
    #     "group": [group], # Group name
    #     "num_worlds": [800],
    #     "resample_scenes": [1], # Yes
    #     "k_unique_scenes": [1000], # Sample in batches of 500
    #     "resample_interval": [2_000_000],
    #     "total_timesteps": [3_000_000_000],
    #     "resample_dataset_size": [1000],
    #     "batch_size": [262_144, 524_288],
    #     "minibatch_size": [16_384],
    #     "update_epochs": [2, 4, 5],
    #     "ent_coef": [0.0001, 0.001, 0.003],
    #     "learning_rate": [1e-4, 3e-4],
    #     "gamma": [0.99],
    #     "render": [0],
    # }

    # save_script(
    #     file_path="examples/experimental/sbatch_scripts/",
    #     filename=f"sbatch_{group}.sh",
    #     fields=fields,
    #     params=hyperparams,
    # )



