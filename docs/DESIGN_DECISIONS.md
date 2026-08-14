# Design Decisions

This document records the key design decisions behind the benchmarking framework, with rationale for each choice. For the full methodology and results, see the [dissertation](TODO).

---

## 1. Model Selection (8 Models, 3 Families)

We selected eight models spanning three architectural families to test whether performance patterns hold across fundamentally different learning paradigms.

**Tree-based ensembles (4):** Random Forest (bagging baseline), XGBoost (second-order gradients), LightGBM (histogram-based, fast on large data), CatBoost (ordered boosting with built-in overfitting detection).

**Deep learning (2):** 1D-CNN (treats the feature vector as a 1D signal to capture local patterns), BiLSTM (bidirectional sequence modeling for feature dependencies). Added per committee feedback to include non-tree, non-transformer deep learning.

**Transformers (2):** FT-Transformer (Feature Tokenizer + Transformer for continuous tabular data), SAINT (Self-Attention and Intersample Attention for tabular data). FT-Transformer was reimplemented from scratch rather than using an external library, for full control over the training process.

**Why not hybrid models?** The dissertation's contribution is a unified benchmarking framework. Including hybrid models with fundamentally different data pipelines (e.g., GCN-Transformer requiring k-NN graph construction) would break the standardized comparison across all eight models.

---

## 2. Classification Tasks

| Task | Classes | Description |
|------|---------|-------------|
| Binary (Task 2) | 2 | Benign vs. Attack |
| Attack Family (Task 6) | 6 | Benign + DDoS + DoS + Recon + MQTT + Spoofing |
| Individual Attack (Task 19) | 19 | Benign + 18 individual attack types |

Task 19 is evaluated only on CIC-IoMT-2024 (the primary dataset). Cross-dataset generalization uses Task 2 and Task 6 only, because the 19-class labels are specific to CIC-IoMT-2024's testbed and have no valid mapping to other datasets.

---

## 3. Preprocessing

**Feature scaling:** RobustScaler (median/IQR normalization). CICFlowMeter features contain extreme outliers from DDoS bursts. MinMaxScaler compresses 99.9% of normal data into a tiny range when a single extreme flow sets the max. StandardScaler's mean and standard deviation are similarly distorted. RobustScaler preserves the normal-range variance while keeping outliers visible as useful signal.

**Class imbalance:** SMOTEENN (SMOTE + Edited Nearest Neighbors). SMOTE generates synthetic minority-class samples. ENN then removes any sample (original or synthetic) whose class disagrees with a majority of its k nearest neighbors, producing cleaner decision boundaries. Applied to training data only; validation and test sets retain the natural imbalanced distribution.

**Feature selection:** Mutual Information with 90% cumulative thresholding. Features are ranked by MI score and those accounting for 90% of total MI are retained. This is model-agnostic (unlike Random Forest importance) and adaptive (binary classification may retain fewer features than 19-class). Transformers always receive the full 76-feature set because our conference paper showed a 64-point F1 drop when feature selection was applied to transformer architectures.

**Validation:** 15% stratified holdout from the training set, created before SMOTEENN is applied. Used for both Optuna hyperparameter optimization and early stopping.

---

## 4. Hyperparameter Optimization

Optuna with TPE sampler, maximizing weighted F1 on the validation set. Trial budgets: 50 trials for tree-based and deep learning models, 100 trials for transformers (larger search space, more sensitive to HPO). SQLite storage per model enables crash recovery. Our conference paper found transformers especially sensitive to insufficient HPO, motivating the higher trial count.

---

## 5. Training

**Early stopping:** Patience of 10 epochs for neural networks and transformers. Transformers use CosineAnnealingWarmRestarts (patience must exceed the restart cycle). Deep learning models use ReduceLROnPlateau (the model needs additional epochs after a learning rate reduction to respond).

**Gradient clipping:** Applied to deep learning and transformer models to prevent gradient explosion.

**Modular execution:** Each (model, task, dataset) combination runs as an independent SLURM job. No job depends on another. Primary training: 24 jobs (8 models x 3 tasks). Cross-dataset: 48 jobs (8 models x 2 tasks x 3 datasets).

---

## 6. Evaluation

**Performance metrics:** Weighted F1 (primary), accuracy, precision, recall (all weighted), per-class F1, confusion matrices.

**Efficiency metrics:** Training time, inference latency (model-only and end-to-end with preprocessing), batch throughput, parameter count, peak memory (training and inference separately). Training vs. inference memory are reported separately because IoMT edge devices are memory-constrained and inference memory is the deployment-relevant number.

**Explainability (two-stage):** Stage 1 runs lightweight SHAP global feature importance on all 8 models. Stage 2 runs full SHAP + LIME analysis on the top-performing model per family (3 models total), producing summary plots, dependence plots, and local instance-level explanations.

**MCDM composite score:** Final Score = 0.60 x Performance + 0.25 x Efficiency + 0.15 x Explainability. All sub-scores are min-max normalized within each task. This multi-criteria ranking avoids selecting a model solely on F1 while ignoring latency or interpretability.

**Bootstrap confidence intervals:** 1,000 bootstrap resamples, 95% CIs for all metrics. Chosen over k-fold cross-validation because (a) k-fold is computationally prohibitive at 6.8-39.3M training samples, (b) with test sets of 680K-3.9M samples any difference would reach statistical significance making p-values uninformative, and (c) CIs show effect size and precision directly.

---

## 7. Cross-Dataset Generalization

**Strategy:** Train on CIC-IoMT-2024, evaluate zero-shot transfer on three external datasets. The entire cross-dataset target is used as the test set (no train/test split of the target) because the target represents a completely unseen network environment.

**Feature alignment:** All four datasets were extracted with Java CICFlowMeter v4.0, producing 76 identical ML-usable features. CIC-BoT-IoT and CIC-ToN-IoT use abbreviated column names (48 of 76 differ); the data loader includes a rename mapping. See `docs/CICFlowMeter_Extraction_Documentation.md` for the re-extraction process.

**Shared-class evaluation:** For Task 6, metrics are computed on the intersection of shared attack families. Flows belonging to novel classes in the target (e.g., Ransomware in CIC-ToN-IoT) are excluded from primary F1 but analyzed separately to characterize how models handle unseen attack types (novel-class absorption).

| Target Dataset | Shared Families | Novel Classes |
|----------------|-----------------|---------------|
| CIC-BoT-IoT | Benign, DDoS, DoS, Recon | Theft |
| CIC-IoT-DIAD-2024 | Benign, DDoS, DoS, Recon, Spoofing | Brute Force, Mirai, Web |
| CIC-ToN-IoT | Benign, DDoS, DoS, Recon | Backdoor, Injection, MITM, Password, Ransomware, XSS |

---

## 8. CIC-IoMT-2024 Re-Extraction

The original CIC-IoMT-2024 dataset ships with DPKT-extracted CSVs (39 packet-window features). All three cross-dataset targets use CICFlowMeter extraction (76 flow-level features). This feature mismatch made cross-dataset transfer impossible. We re-extracted all 72 CIC-IoMT-2024 PCAP files using Java CICFlowMeter v4.0 on an HPC cluster (managed by South Dakota State University), producing 6,732,090 flows with 76 ML-usable features aligned across all four datasets.

See `docs/CICFlowMeter_Extraction_Documentation.md` for the full process.
