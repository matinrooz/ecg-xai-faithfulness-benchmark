# Benchmarking Explainable AI Methods for ECG Classification Using Perturbation-Based Faithfulness Evaluation

The code for my research project at the University of Basel. I trained two models on the
PTB-XL ECG dataset (12 leads, 100 Hz) and then checked whether explanation methods really
point to the parts of the signal the models use.

- **Tasks:** five diagnostic classes (CD, HYP, MI, NORM, STTC), and myocardial infarction vs.
  normal.
- **Models:** a CNN and a CNN-BiLSTM with attention, each trained with 10 seeds.
- **Explanation methods:** Integrated Gradients, SmoothGrad-IG and Deep Taylor Decomposition.
- **How faithfulness is measured:** the ECG regions an explanation marks as important are
  removed step by step (set to zero or interpolated), and the drop in the model's performance
  is compared with removing random regions. A faithful explanation should hurt the model
  faster than random.
- **Statistics:** bootstrap confidence intervals, Wilcoxon signed-rank tests and Bland-Altman
  agreement between methods and perturbation strategies.

## What is in the folder

```
ecg_xai_benchmark/
├── requirements.txt
├── data/processed/   # only meta.json, the arrays are rebuilt by the preprocessing scripts
├── src/              # preprocessing, training, evaluation and plotting
│   └── xai_benchmark/   # explanation methods and the perturbation benchmark
├── result/           # classification results (metrics, tables, per-seed predictions)
└── results/          # benchmark results (reports, curves, Bland-Altman)
```

The processed PTB-XL arrays and the trained checkpoints are not included because they are
large. Both can be rebuilt: the arrays from the public PTB-XL data with the preprocessing
scripts, the checkpoints by retraining with the fixed seeds. All the result files with the
reported numbers are included, so you can look at them without running anything.

## Run

Python 3.10.

```bash
python3 -m venv ecg-env
source ecg-env/bin/activate
pip install -r requirements.txt

cd src
# 1. preprocess PTB-XL (download it from PhysioNet first)
python preprocess_binary.py --data_root <path-to-PTB-XL> --out_dir ../data/processed/ptbxl_lr_binary_norm_mi

# 2. train one model (repeat for each seed)
python train_cnn.py --data_dir ../data/processed/ptbxl_lr_binary_norm_mi \
  --out_dir ../result/binary_experiment/cnn_deeper/cnn_deeper_binary_seed_42 \
  --arch deeper --num_classes 2 --seed 42 --device cuda

# 3. run the explanation benchmark
python -m xai_benchmark.main --data_dir ../data/processed/ptbxl_lr_binary_norm_mi \
  --cnn_ckpt <cnn best.pt> --lstm_ckpt <cnn-lstm best.pt> \
  --out_dir ../results --num_classes 2 --target_class 1
```

I ran the training and the benchmark on the sciCORE cluster at the University of Basel, with
a GPU. Everything also runs on CPU with `--device cpu`, just slower.

All commands with their full settings (both tasks, both models, the ten seeds, the
Bland-Altman analysis and every figure and table) are in [`USAGE.md`](USAGE.md).
