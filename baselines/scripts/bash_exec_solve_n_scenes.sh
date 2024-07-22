# !/bin/sh

# Run PPO script
/scratch/dc4971/.conda/gpudrive/bin/python baselines/ippo/run_sb3_ppo.py --lr=1e-4 --num_worlds=100 --n_epochs=2
