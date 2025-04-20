## Waymo Open Sim Agent Challenge (WOSAC) evaluation


## Requirements
Prerequisite 
```
pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.4
pip install --no-deps git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax
```

## Dataset
Extract TF example from raw waymo TF example dataset using
```
python examples/eval/extract_dataset.py --data_dir XXXX --save_dir xxxx --dataset [train/val/val_interactive]
```

e.g.

```
python examples/eval/extract_dataset.py --data_dir data/raw --save_dir data/processed/wosac_val --dataset all
```


## Evaluation
Run eval with
```
python wosac_eval.py
```