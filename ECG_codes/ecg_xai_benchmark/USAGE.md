# Usage

All commands, from raw data to every figure and table. The overview is in the [README](README.md).

## 1. Data preparation

The raw **PTB-XL** dataset (PhysioNet, 12-lead) must be downloaded separately and passed to the
preprocessing scripts via `--data_root`. The scripts write the processed arrays to
`data/processed/`:

```bash
cd src
python preprocess.py        --data_root <path-to-raw-PTB-XL> --out_dir ../data/processed/ptbxl_lr_5class
python preprocess_binary.py --data_root <path-to-raw-PTB-XL> --out_dir ../data/processed/ptbxl_lr_binary_norm_mi
```

Both scripts additionally accept `--max_train`, `--max_val`, `--max_test` (sample caps) and the
band-pass parameters `--low`, `--high`, `--order`. The processed folders contain the
`X_{train,val,test}.npy`, `y_{train,val,test}.npy`, and `meta.json` files used downstream.


---

## 2. Running the classification experiments

All commands are run from `src/`. Seeds used for the final results: `42, 123, 777, 2024, 9999,
47, 48, 49, 50, 51`.

### 2.1 Five-class, CNN (Deeper CNN)
```bash
python train_cnn.py --data_dir ../data/processed/ptbxl_lr_5class \
  --out_dir ../result/cnn_final/deeper_cnn_seed_42 \
  --arch deeper --num_classes 5 --epochs 20 --batch_size 128 --lr 1e-3 \
  --weight_decay 1e-4 --dropout 0.2 --grad_clip 1.0 --seed 42 --device cuda
```

### 2.2 Five-class, CNN–BiLSTM with Attention
```bash
python cnn_lstm_attn.py --data_dir ../data/processed/ptbxl_lr_5class \
  --out_dir ../result/cnn_lstm_attn_weighted_final/cnn_lstm_attn_weighted_42 \
  --epochs 20 --batch_size 128 --lr 1e-3 --weight_decay 1e-4 --dropout 0.3 \
  --lstm_hidden 64 --attn_hidden 64 --use_class_weights --grad_clip 1.0 --seed 42 --device cuda
```

### 2.3 Binary, CNN (Deeper CNN)
```bash
python train_cnn.py --data_dir ../data/processed/ptbxl_lr_binary_norm_mi \
  --out_dir ../result/binary_experiment/cnn_deeper/cnn_deeper_binary_seed_42 \
  --arch deeper --num_classes 2 --epochs 20 --batch_size 128 --lr 1e-3 \
  --weight_decay 1e-4 --dropout 0.2 --grad_clip 1.0 --seed 42 --device cuda
```

### 2.4 Binary, CNN–BiLSTM with Attention (unweighted)
```bash
python cnn_lstm_attn.py --data_dir ../data/processed/ptbxl_lr_binary_norm_mi \
  --out_dir ../result/binary_experiment/cnn_lstm_binary/cnn_lstm_binary_seed_42 \
  --epochs 20 --batch_size 128 --lr 1e-3 --weight_decay 1e-4 --dropout 0.3 \
  --lstm_hidden 64 --attn_hidden 64 --grad_clip 1.0 --seed 42 --device cuda
```

Repeat each command for every seed. Evaluation (writes `metrics_test.json` next to the checkpoint):
```bash
python eval_cnn.py      --data_dir <data> --ckpt <path>/best.pt --split test --num_classes {2|5} --arch deeper --save_txt
python eval_cnn_lstm.py --data_dir <data> --ckpt <path>/best.pt --split test --num_classes {2|5} --dropout 0.3 --lstm_hidden 64 --attn_hidden 64 --save_txt
```
*(`train_cnn.py` takes `--num_classes`; `cnn_lstm_attn.py` infers the class count from the data.)*

---

## 3. Multi-seed evaluation

The final classification results are aggregated across the ten seeds. For each task, the
`export_*` script runs inference and saves the per-seed probabilities, and the `plot_*` scripts
then produce the metric boxplots, the mean ROC and Precision–Recall curves, and the summary tables.

```bash
# five-class
python export_5class_10seed_probs.py
python plot_figure25_2x2_10seeds.py
python plot_5class_10seed_roc_pr.py

# binary
python export_binary_10seed_probs.py
python plot_binary_10seed.py
```
The export scripts read the checkpoints in `result/cnn_final/`, `result/cnn_lstm_attn_weighted_final/`,
and `result/binary_experiment/`, and write predictions and figures under `result/figures_5class_10seed/`
and `result/figures_binary_10seed/`.

---

## 4. Running the XAI perturbation benchmark

The benchmark is driven by `xai_benchmark/main.py`. It computes IG, SmoothGrad-IG, and DTD
attributions, ranks ECG regions, perturbs them stepwise (zero-out and interpolation) at both
segment- and timestep-level granularity, and records the perturbation AUC against a matched
random-order baseline.

```bash
python -m xai_benchmark.main \
  --data_dir ../data/processed/ptbxl_lr_binary_norm_mi \
  --cnn_ckpt  ../result/binary_experiment/cnn_deeper/cnn_deeper_binary_seed_42/best.pt \
  --lstm_ckpt ../result/binary_experiment/cnn_lstm_binary/cnn_lstm_binary_seed_42/best.pt \
  --out_dir  ../results --num_classes 2 --target_class 1 --ig_steps 50
```

Subsets you can select:
```
--methods        ig smoothgrad_ig dtd
--perturbations  zero_out interpolation
--granularities  segment timestep
--models_to_run  cnn cnn_lstm
```
Additional arguments include `--sg_samples`, `--edge_mask`, `--max_samples`, `--max_norm_samples`,
`--synth_method`, `--seed`, `--random_seed`, `--split`.

**Segment-level is the primary benchmark in the report; the timestep-level benchmark is
supplementary.** Both are produced by the same command via `--granularities`.

---

## 5. Bland–Altman analysis

```bash
python -m xai_benchmark.bland_altman \
  --ig_results_dir ../results/IG \
  --smoothgrad_results_dir ../results/smoothgrad \
  --dtd_results_dir ../results/dtd \
  --out_dir ../results/bland_altman

python -m xai_benchmark.bland_altman_mi_only     # MI-only comparisons
python -m xai_benchmark.plot_method_curves_grid  # perturbation curve-grid figure
```
`bland_altman.py` produces the agreement statistics between attribution methods, perturbation
strategies, and architectures; `bland_altman_mi_only.py` restricts the interpolation-versus-zero-out
comparison to MI samples. (`bland_altman_mi_only.py` and `plot_method_curves_grid.py` use fixed input and output paths.)

---

## 6. Attribution heatmaps

```bash
python plot_bandpass_example.py                    # band-pass illustration (--out to set filename)
python -m xai_benchmark.plot_2s_crop_attribution   # 2-second cropped IG/SG/DTD heatmaps (sample idx 31)
```

---

## 7. Reproducing the figures and tables

- `plot_bandpass_example.py`: the band-pass filtering figure
- `export_5class_10seed_probs.py`, `plot_figure25_2x2_10seeds.py`, `plot_5class_10seed_roc_pr.py`: Table 3, Figures 25-26
- `export_binary_10seed_probs.py`, `plot_binary_10seed.py`: Table 4, Figures 27-28
- `xai_benchmark/main.py`: Table 5, and Tables 2 and 6 with `--check_completeness`
- `xai_benchmark/bland_altman.py`, `bland_altman_mi_only.py`: Table 7
- `xai_benchmark/plot_method_curves_grid.py`: Figure 34
- `xai_benchmark/plot_2s_crop_attribution.py`: the attribution heatmaps

Table 11 (confusion matrices) is reported from the seed-42 binary inference and has **no dedicated
script** in the repository.

---

## 8. Running on sciCORE

The results were produced on the sciCORE HPC cluster using an interactive GPU session and the
`ecg-env` virtual environment. The workflow was:

```bash
cd ecg_xai_benchmark/src
source ~/ecg-env/bin/activate
python -m xai_benchmark.main ...        # (see section 4)
```

There are no SLURM batch scripts in the repository, I ran the jobs in an interactive GPU session. Scripts accept
`--device cuda` on a GPU node and fall back to `--device cpu` otherwise.

---

## 9. Saved outputs

The repository includes the following (see the README for what is not included):

- **Model checkpoints** (`best.pt`) are **not bundled** here to reduce size; regenerate
  them by retraining with the fixed seeds (section 2). Their original locations were
  `result/cnn_final/`, `result/cnn_lstm_attn_weighted_final/`, and `result/binary_experiment/`.
- **Per-seed prediction arrays** (`.npy`) under `result/figures_5class_10seed/predictions/` and
  `result/figures_binary_10seed/predictions/` **are included**, they let the classification
  figures be regenerated without retraining.
- **Classification tables/summaries** (`.tex`, `.csv`) under `result/figures_5class_10seed/` and
  `result/figures_binary_10seed/` are included; the corresponding **summary PNG figures are not
  bundled** and regenerate from the prediction arrays via the `plot_*` scripts (section 3).
- **Perturbation benchmark reports** (`final_report.txt`, `summary.csv`) under
  `results/{IG,smoothgrad,dtd}/`.
- **IG completeness reports** under `result/completeness/`.
- **Bland–Altman reports** under `results/bland_altman/` and `results/bland_altman_mi_only/`.
- **Perturbation curve-grid figures** under `results/figures_method_curve_grid/`.

---

## 10. Reproducibility notes

- **Splits:** training uses the processed `train`/`val` arrays; all reported metrics are computed
  on the held-out `test` split.
- **Seeds:** each run sets its seed explicitly; the ten-seed list is
  `42, 123, 777, 2024, 9999, 47, 48, 49, 50, 51`. Classification results (Tables 3–4, Figures 25–28)
  are aggregated across these ten seeds.
- **Checkpoint naming:** `<experiment>_seed_<seed>/best.pt`. For the binary Deeper CNN, seeds
  `42,123,777,2024,9999` are stored under `binary_experiment/cnn_deeper/` and seeds `47–51` under
  `binary_experiment/cnn_binary/` (both are the deeper architecture).
- **`result/` vs `results/`:** `result/` holds classification checkpoints and classification
  figures/tables; `results/` holds the XAI benchmark outputs (attributions, perturbation curves,
  Bland–Altman, completeness).
- **Representative-seed analyses:** the XAI perturbation and attribution analyses use the seed-42
  binary checkpoints as the representative models.

---

