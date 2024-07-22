'''
A script for generating SLURM submission scripts which sweep parameters.
'''
import numpy as np
import os
import re
from typing import Dict

LOG_FOLDER = 'experiments/slurm/logs/'

# Default SLURM fields
DEFAULT_SLURM_FIELDS = {
    'num_nodes': 1,
    'num_cpus': 1,
    'num_gpus': 1,
    'gpu_type': None,  # --gres=gpu:1:rtx8000; logic: if gpu_type in supported list, add to end. If not supported list, throw exception, and if not provided, don't add GPU type
    'memory': 10,
    'memory_unit': 'GB',
    'time_d': 0,'time_h': 0, 'time_m': 0, 'time_s': 0,  
    'max_sim_jobs': None,
    'output': f'{LOG_FOLDER}output_%A_%a.txt',
    'error': f'{LOG_FOLDER}error_%A_%a.txt',
}

# a template for the entire submit script
# (bash braces must be escaped by doubling: $var = ${{var}})
# num_jobs, param_arr_init, param_val_assign and param_list are special fields

TEMPLATE_SBATCH = '''
#!/bin/bash

#SBATCH --array=0-{num_jobs}%{max_sim_jobs}
#SBATCH --job-name={job_name}
#SBATCH --output={output}
#SBATCH --error={error}
#SBATCH --mem={memory}{memory_unit}
#SBATCH --time={time_d}-{time_h}:{time_m}:{time_s}
#SBATCH --nodes={num_nodes}
#SBATCH --cpus-per-task={num_cpus}
#SBATCH --gres=gpu:a100:{num_gpus}

SINGULARITY_IMAGE=hpc/nocturne.sif
OVERLAY_FILE=hpc/overlay-15GB-500K.ext3

singularity exec --nv --overlay /scratch/dc4971/gpudrive/hpc/overlay-15GB-500K.ext3:ro" \
    "/scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif" \
    /bin/bash baselines/scripts/bash_exec.sh "${{SLURM_ARRAY_TASK_ID}}"
echo "Successfully launched image."
'''.strip()

TEMPLATE_BASH = '''
# !/bin/sh
{param_arr_init}

trial=${{SLURM_ARRAY_TASK_ID}}
{param_val_assign}

/scratch/dc4971/.conda/gpudrive/bin/python baselines/ippo/run_sb3_ppo.py {param_cli_list}
'''

# functions for making bash expressions
# bash braces are escaped by doubling

def _mth(exp):
    return '$(( %s ))' % exp
def _len(arr):
    return '${{#%s[@]}}' % arr
def _get(arr, elem):
    return '${{%s[%s]}}' % (arr, elem)
def _eq(var, val):
    return '%s=%s' % (var, val)
def _op(a, op, b):
    return _mth('%s %s %s' % (a, op, b))
def _arr(arr):
    return '( %s )' % ' '.join(map(str, arr))
def _seq(a, b, step):
    return '($( seq %d %d %d ))' % (a, step, b)
def _var(var):
    return '${%s}' % var
def _cli_var(var):
    tmp = f'--{var}'.replace('_', '_')
    return f'{tmp}=${{{var}}}'


# Templates for param array construction and element access
PARAM_ARR = '{param}_values'
PARAM_EXPRS = {
    'param_arr_init':
        _eq(PARAM_ARR, '{values}'),
    'param_val_assign': {
        'assign':
            _eq('{param}', _get(PARAM_ARR, _op('trial','%',_len(PARAM_ARR)))),
        'increment':
            _eq('trial', _op('trial', '/', _len(PARAM_ARR)))
    }
}
def _to_bash(obj):
    if isinstance(obj, range):
        return _seq(obj.start, obj.stop - 1, obj.step)
    if isinstance(obj, list) or isinstance(obj, tuple):
        return _arr(obj)
    raise ValueError('Unknown object type %s' % type(obj).__name__)


def _get_params_bash(params, values):
    # builds bash code to perform the equivalent of
    '''
    def get_inds(params, ind):
        inds = []
        for length in map(len, params):
            inds.append(ind % length)
            ind //= length
        return inds[::-1]
    '''
    # get lines of bash code for creating/accessing param arrays
    init_lines = []
    assign_lines = []
    init_temp = PARAM_EXPRS['param_arr_init']
    assign_temps = PARAM_EXPRS['param_val_assign']

    for param, vals in zip(params, values):
        init_lines.append(
            init_temp.format(param=param, values=_to_bash(vals)))
        assign_lines.append(
            assign_temps['assign'].format(param=param))
        assign_lines.append(
            assign_temps['increment'].format(param=param))

    # remove superfluous final trial reassign
    assign_lines.pop()

    return init_lines, assign_lines


def get_scripts(fields: Dict = DEFAULT_SLURM_FIELDS, params: Dict = {}, param_order=None, run_script=None):
    '''
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
    '''

    # check arguments have correct type
    assert isinstance(fields, dict)
    assert isinstance(params, dict)
    assert (isinstance(param_order, list) or
            isinstance(param_order, tuple) or
            param_order==None)
    if param_order == None:
        param_order = list(params.keys())

    # check each field appears in the template
    for field in fields:
        if ('{%s}' % field) not in TEMPLATE_SBATCH:
            raise ValueError('passed field %s unused in template' % field)

    # calculate total number of jobs (minus 1; SLURM is inclusive)
    num_jobs = 1
    for vals in params.values():
        num_jobs *= len(vals)
    num_jobs -= 1

    # get bash code for param sweeping
    init_lines, assign_lines = _get_params_bash(
        param_order, [params[key] for key in param_order])

    # build template substitutions (overriding defaults)
    subs = {
        'param_arr_init': '\n'.join(init_lines),
        'param_val_assign': '\n'.join(assign_lines),
        'param_cli_list': ' '.join(map(_cli_var, param_order)),
        'num_jobs': num_jobs
    }

    for key, val in DEFAULT_SLURM_FIELDS.items():
        subs[key] = val
    for key, val in fields.items():
        subs[key] = val
    if 'job_name' not in subs:
        subs['job_name'] = "my_job"

    return TEMPLATE_SBATCH.format(**subs), TEMPLATE_BASH.format(**subs)


def save_scripts(sbatch_filename, bash_filename, file_path, run_script, fields, params, param_order=None):
    '''
    creates and writes to file a SLURM submission script using the passed
    fields and which creates an array of jobs which sweep the given params

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
    '''
    
    # Create scripts
    sbatch_script, bash_script = get_scripts(fields, params, param_order, run_script)

    if not file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path+sbatch_filename, 'w') as file:
        file.write(sbatch_script)

    with open(file_path+bash_filename, 'w') as file:
        file.write(bash_script)


if __name__ == '__main__':

    SWEEP_NAME = 'paper_fig'

    # Define SBATCH params
    fields = {
        'time_h': 5, # Max time per job
        'num_gpus': 1, # GPUs per job 
        'max_sim_jobs': 50, # Max jobs at the same time
        'job_name': SWEEP_NAME,
    }

    params = {
        'sweep_name': [SWEEP_NAME], # Project name
        'lr': [3e-4, 1e-3],
        'num_worlds': [50, 100, 150],
        'n_epochs': [10, 5, 2],
        'num_minibatches': [4, 5, 10], 
    }
    
    save_scripts(
        sbatch_filename=f"sbatch_{SWEEP_NAME}.sh",
        bash_filename="bash_exec.sh", #NOTE: don't change this name
        file_path="baselines/scripts/",
        run_script="baselines/ippo/run_sb3_ppo.py",
        fields=fields,
        params=params,
    )