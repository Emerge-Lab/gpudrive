from pathlib import Path

GPU_DRIVE_DATA_ROOT = Path('/'.join(__path__[0].split('/')[:-1]))
GPU_DRIVE_DATA_DIR = GPU_DRIVE_DATA_ROOT / 'data/processed/training'
