# A Unified Benchmarking Framework for Intrusion Detection in the Internet of Medical Things

This repository contains the complete codebase for the doctoral dissertation:

> **A Unified Benchmarking Framework for Intrusion Detection in the Internet of Medical Things**
> Oswald Adohinzin, Dakota State University, 2026

**Dissertation:** Dissertation link forthcoming

The framework evaluates **8 machine learning models** across **3 architectural families** on the CIC-IoMT-2024 dataset, measuring performance, computational efficiency, explainability, and cross-dataset generalization.

---

## Models

| Family | Models |
|--------|--------|
| Tree-based | Random Forest, XGBoost, LightGBM, CatBoost |
| Deep Learning | 1D-CNN, BiLSTM |
| Transformers | FT-Transformer, SAINT |

## Classification Tasks

| Task | Classes | Scope |
|------|---------|-------|
| Binary (Task 2) | 2 | Benign vs. Attack |
| Attack Family (Task 6) | 6 | Benign + 5 attack families |
| Individual Attack (Task 19) | 19 | Benign + 18 attack types |

## Evaluation Dimensions

1. **Performance** -- Weighted F1, accuracy, precision, recall, per-class metrics, confusion matrices
2. **Efficiency** -- Inference latency, throughput, memory footprint, training time, parameter count
3. **Explainability** -- SHAP global feature importance (all models) + SHAP/LIME deep analysis (top model per family)
4. **Cross-dataset generalization** -- Zero-shot transfer from CIC-IoMT-2024 to CIC-BoT-IoT, CIC-IoT-DIAD-2024, and CIC-ToN-IoT

A **Multi-Criteria Decision Making (MCDM)** composite score ranks models: `Final = 0.60 * Performance + 0.25 * Efficiency + 0.15 * Explainability`.

---

## Repository Structure

```
.
├── config/
│   └── config.yaml              # Central configuration (datasets, tasks, models, HPO)
├── preprocessing/
│   ├── data_loader.py           # Dataset loading with column standardization
│   ├── preprocessing.py         # Cleaning, scaling, SMOTEENN, caching
│   ├── feature_selection.py     # Mutual Information with cumulative thresholding
│   └── feature_alignment_verification.py
├── models/
│   ├── base.py                  # Abstract base class
│   ├── tree_based/              # RF, XGBoost, LightGBM, CatBoost
│   ├── deep_learning/           # CNN1D, BiLSTM
│   └── transformers/            # FT-Transformer, SAINT
├── training/
│   ├── train.py                 # Unified training loop with Optuna HPO
│   ├── early_stopping.py        # Patience-based early stopping
│   └── pytorch_utils.py         # Device management, gradient clipping
├── evaluation/
│   ├── metrics.py               # Performance + efficiency metrics + bootstrap CIs
│   ├── model_selection.py       # MCDM composite scoring
│   ├── cross_dataset.py         # Cross-dataset transfer evaluation
│   └── explainability/          # SHAP and LIME analysis
├── scripts/
│   ├── run_single.py            # Run one (model, task, dataset) combination
│   ├── run_all.py               # Orchestrate all primary training jobs
│   ├── run_cross_dataset.py     # Run cross-dataset generalization
│   ├── run_model_selection.py   # Compute MCDM rankings
│   ├── run_preprocessing.py     # Preprocess and cache data
│   ├── generate_report.py       # Generate leaderboard CSVs
│   └── slurm_*.sh               # SLURM job scripts for HPC execution
├── results/                     # Per-model JSON results (performance, efficiency, CI)
├── reports/
│   ├── leaderboards/            # Ranked model comparisons per task (CSV)
│   └── model_selection/         # MCDM composite scores per task (CSV)
├── docs/
│   ├── DESIGN_DECISIONS.md      # Rationale for all major design choices
│   ├── CICFlowMeter_Extraction_Documentation.md
│   ├── Feature_Alignment_Report.md
│   └── EXPLAINABILITY_CLUSTER_INSTRUCTIONS.md
├── requirements.txt
├── LICENSE                      # MIT
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for deep learning and transformer models)
- 64GB+ RAM (SMOTEENN on the full dataset is memory-intensive)

### Installation

```bash
git clone https://github.com/adoregas95/A-Unified-Benchmarking-Framework-for-Intrusion-Detection-in-the-IoMT.git
cd A-Unified-Benchmarking-Framework-for-Intrusion-Detection-in-the-IoMT
pip install -r requirements.txt
```

### Data

The datasets are not included in this repository. Download them from their original sources:

- **CIC-IoMT-2024**: https://www.unb.ca/cic/datasets/iomt-dataset-2024.html
- **CIC-BoT-IoT**: https://www.unb.ca/cic/datasets/botiot.html
- **CIC-IoT-DIAD-2024**: https://www.unb.ca/cic/datasets/iot-diad-2024.html
- **CIC-ToN-IoT**: https://www.unb.ca/cic/datasets/toniot.html

Place each dataset under `data/` following the paths in `config/config.yaml`.

**Important:** CIC-IoMT-2024 ships with DPKT-extracted CSVs, but this framework requires CICFlowMeter-extracted features for cross-dataset compatibility. See `docs/CICFlowMeter_Extraction_Documentation.md` for the re-extraction process.

---

## Usage

### Single Model Run

```bash
python scripts/run_single.py --model XGBoost --task 6 --dataset CICIoMT2024
```

### Full Pipeline

```bash
# 1. Preprocess and cache data for all tasks
python scripts/run_preprocessing.py

# 2. Train all models on all tasks
python scripts/run_all.py

# 3. Cross-dataset generalization
python scripts/run_cross_dataset.py

# 4. MCDM model selection
python scripts/run_model_selection.py

# 5. Generate leaderboard reports
python scripts/generate_report.py
```

### HPC (SLURM)

SLURM job scripts are provided in `scripts/slurm_*.sh`. Each script submits independent jobs for parallel execution on an HPC cluster. Edit the resource requests (partition, memory, walltime) to match your cluster configuration.

```bash
sbatch scripts/slurm_train_tree.sh      # All tree-based models
sbatch scripts/slurm_train_dl.sh        # CNN1D, BiLSTM
sbatch scripts/slurm_train_transformer.sh  # FT-Transformer, SAINT
```

---

## Results

Results are stored as JSON files under `results/{dataset}/task_{N}/{model}/results.json`. Each file contains performance metrics, efficiency measurements, bootstrap confidence intervals, and the best hyperparameters found by Optuna.

The MCDM composite scores are in `results/model_selection_results.json`.

Leaderboard summaries (CSV) are in `reports/leaderboards/` and `reports/model_selection/`.

---

## Key Design Decisions

See `docs/DESIGN_DECISIONS.md` for the rationale behind choices such as:

- RobustScaler over MinMaxScaler/StandardScaler for CICFlowMeter data
- SMOTEENN over plain SMOTE for class imbalance handling
- Mutual Information cumulative thresholding over fixed feature counts
- Full feature set for transformers (no feature selection)
- Bootstrap CIs over k-fold cross-validation at this data scale
- MCDM weighting scheme (0.60 / 0.25 / 0.15)

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@phdthesis{adohinzin2026unified,
  title   = {A Unified Benchmarking Framework for Intrusion Detection
             in the Internet of Medical Things},
  author  = {Adohinzin, Oswald},
  school  = {Dakota State University},
  year    = {2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
