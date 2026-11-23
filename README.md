# ecg-xai-faithfulness-benchmark

**What:** My MSc thesis (University of Basel, 2026): a benchmark that checks whether explanation methods for ECG classifiers point to the parts of the signal the model really uses, by perturbing those parts.  
**Result:** Mean test ROC-AUC 0.972 ± 0.002 over 10 seeds for myocardial infarction vs. normal; Deep Taylor Decomposition gave the most faithful explanations, and the perturbation strategy changed the result more than the choice of explanation method.  
**Stack:** Python, PyTorch, PTB-XL, SLURM

The code and the full instructions are in [`ECG_codes/master_thesis`](ECG_codes/master_thesis/README.md), the thesis is `Master_Thesis Matin Roozbahani .pdf`.
