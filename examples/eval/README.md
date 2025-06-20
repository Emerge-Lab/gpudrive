## Waymo Open Sim Agent Challenge (WOSAC) evaluation


## Requirements
Prerequisite to run the eval
```
pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.6
```

Requirement to process the data
```
pip install --no-deps git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax
```

You also need
```
pip install tensorflow==2.13.0
pip install scikit-learn
pip install tensorflow-probability==0.21.0
pip install --upgrade typing-extensions
```


## Dataset
Extract TF example from raw waymo TF example dataset using
```
python examples/eval/extract_dataset.py --data_dir XXXX --save_dir xxxx --dataset [train/val/val_interactive]
```

e.g.

```
python examples/eval/extract_dataset.py --data_dir data/raw --save_dir data/processed/wosac --dataset all
```


## Evaluation
Run eval with
```
python run_wosac_eval.py
```
