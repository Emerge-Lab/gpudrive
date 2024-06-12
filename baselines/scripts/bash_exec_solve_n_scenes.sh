# !/bin/sh

# Run PPO script
/scratch/dc4971/.conda/gpudrive/bin/python baselines/ippo/run_sb3_ppo.py --batch_size=2048 --lr=1e-3 --train_on_k_unique_scenes=1024 --num_worlds=1024
