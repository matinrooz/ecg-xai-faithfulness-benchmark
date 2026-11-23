# ecg-xai-faithfulness-benchmark

**What:** Research project at the University of Basel (2026): a benchmark that checks whether explanation methods for ECG classifiers point to the parts of the signal the model really uses, by perturbing those parts.  
**Result:** Mean test ROC-AUC 0.972 ± 0.002 over 10 seeds for myocardial infarction vs. normal; Deep Taylor Decomposition gave the most faithful explanations, and the perturbation strategy changed the result more than the choice of explanation method.  
**Stack:** Python, PyTorch, PTB-XL, SLURM

The code and the full instructions are in [`ECG_codes/ecg_xai_benchmark`](ECG_codes/ecg_xai_benchmark/README.md), the full report is [`ECG_XAI_benchmark_report.pdf`](ECG_XAI_benchmark_report.pdf).
