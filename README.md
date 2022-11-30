# ADL22-HW3

## Reproduce 
### Dataset & pre-preprocess
```
python3.8 prepreprocess.py
```
### Training
Models will be stored in `results/50ep_warm`
```
./run_train_summary.sh
```
### Eval
This script will eval every epoch's models (`0-49`).
```
./run_all_infer_eval_summary.sh
```

## Inference & Predict
Predict with my best model and strategy.
```
./run.sh INPUT_JSONL OUTPUT_JSONL
```
