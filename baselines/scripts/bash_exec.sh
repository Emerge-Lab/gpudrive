# !/bin/sh

# Run PPO script
/scratch/dc4971/.conda/gpudrive/bin/python baselines/ippo/run_sb3_ppo.py --seed=42 --n_epochs=2 --batch_size=2048
