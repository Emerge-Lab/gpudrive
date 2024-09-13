
# !/bin/sh
sweep_name_values=( paper_fig )
lr_values=( 0.0003 0.001 )
num_worlds_values=( 50 100 150 )
n_epochs_values=( 10 5 2 )
num_minibatches_values=( 4 5 10 )

trial=${SLURM_ARRAY_TASK_ID}
sweep_name=${sweep_name_values[$(( trial % ${#sweep_name_values[@]} ))]}
trial=$(( trial / ${#sweep_name_values[@]} ))
lr=${lr_values[$(( trial % ${#lr_values[@]} ))]}
trial=$(( trial / ${#lr_values[@]} ))
num_worlds=${num_worlds_values[$(( trial % ${#num_worlds_values[@]} ))]}
trial=$(( trial / ${#num_worlds_values[@]} ))
n_epochs=${n_epochs_values[$(( trial % ${#n_epochs_values[@]} ))]}
trial=$(( trial / ${#n_epochs_values[@]} ))
num_minibatches=${num_minibatches_values[$(( trial % ${#num_minibatches_values[@]} ))]}

/scratch/dc4971/.conda/gpudrive/bin/python baselines/ippo/run_sb3_ppo.py --sweep_name=${sweep_name} --lr=${lr} --num_worlds=${num_worlds} --n_epochs=${n_epochs} --num_minibatches=${num_minibatches}
