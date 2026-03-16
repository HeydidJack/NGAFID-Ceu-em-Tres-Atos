# Aircraft Health Diagnosis via Task Decomposition

Research code for "Balancing Safety and Efficiency in Aircraft Health Diagnosis — A Task Decomposition Framework with Heterogeneous Long-Micro Scale Cascading". This repository provides implementations for heterogeneous two-stage diagnosis on real-world aviation flight data.

## Dataset

**NGAFID General Aviation Dataset**
- **Source**: Zenodo DOI [10.5281/zenodo.6624956](https://doi.org/10.5281/zenodo.6624956)
- **Content**: Cessna 172 fleet, 23-dimensional sensor time-series (1Hz), 28,935 flights (31,000+ hours), 36 unscheduled maintenance categories
- **Partitions**: 
  - `2days` (Subset): 19 classes, 11,446 flights
  - `all_flights` (Overall): 36 classes, 28,935 flights
- **Size**: ~5.4 GB compressed, ~9 GB uncompressed

### Setup

**Automated (Recommended)**:
```bash
python scripts/setup_dataset.py
```

**Manual**:
1. Download from [Zenodo](https://zenodo.org/record/6624956):
   - `2days.tar.gz` (1.1 GB) → `Datasets/2days/`
   - `all_flight.tar.gz` (4.3 GB) → `Datasets/all_flights/`

## Data Preparation

Preprocess raw flight records to model-ready tensors (fixed 2048 timesteps via cubic interpolation):

```bash
# Subset (faster experimentation)
python prepare_diagnosis_data.py --dataset subset --seed 350234

# Full dataset
python prepare_diagnosis_data.py --dataset full --seed 350234
```

**Output**: `ProcessedData/ProcessedData_Diagnosis_DeD_{subset|full}/`
- Stratified 5-fold cross-validation splits
- Fault mapping configurations (ID 0 reserved for healthy)
- Z-score normalization parameters (fit on train only)

## Supported Models

| Model | Architecture | Receptive Field | Use Case |
|-------|-------------|-----------------|----------|
| **Bi-LSTM** | 3-layer Bi-LSTM (h=128) | Sequential | Baseline |
| **InceptionTime** | Parallel conv (10/20/40) | Multi-scale local | End-to-end |
| **MMK Net** | Multi-kernel {1,3,5} CNN + LayerNorm | Restricted local | Fault Classification |
| **ConvTokMHSA** | Conv tokenizer + Global MHSA | Full-sequence | Anomaly Detection |
| **ConvTokSWLA** | Conv tokenizer + Sliding Window Attn | Local window | Comparison baseline |
| **LMSD (Ours)** | ConvTokMHSA → MMK Net (cascaded) | Global + Local | Two-stage Diagnosis |

## Experiments

All experiments use **Stratified 5-Fold Cross-Validation** (physical file-level isolation), repeated 3 rounds with different random seeds, reporting median performance.

### 1. Anomaly Detection (AD)
Binary classification: Healthy vs Anomalous. Uses full dataset without augmentation.

```bash
python exp_ad_detection.py
# Configure in configs/ad_config.py: model_name, data_select_pattern, etc.
```

**Key Results** (from paper Table 3, median of 5-fold×3 rounds):

| Model | Subset ACC | Subset F1 | Subset FNR | Overall ACC | Overall F1 | Overall FNR |
|-------|-----------|-----------|------------|-------------|------------|-------------|
| ConvTokMHSA | **0.7991** | **0.7990** | **0.1634** | **0.7657** | **0.7640** | **0.2499** |
| InceptionTime | 0.7838 | 0.7838 | 0.1962 | 0.7258 | 0.7205 | 0.3512 |
| MMK Net | 0.7210 | 0.7208 | 0.2977 | 0.7090 | 0.7036 | 0.3673 |

**Efficiency** (Overall dataset):
- ConvTokMHSA: ET=13.86s/epoch, TTT=297.70s, IT32=0.01s, MSize=1.54MB
- InceptionTime: ET=160.05s/epoch, TTT=3502.96s

### 2. Fault Classification (FC)
Multi-class fault type recognition (19 or 36 classes). **Trains only on anomalous samples** with proportional replication augmentation (k=3).

```bash
python exp_fc_classification.py
# Configure in configs/fc_config.py
```

**Key Results** (from paper Table 4):

| Model | Subset ACC | Subset F1 | Overall ACC | Overall F1 |
|-------|-----------|-----------|-------------|------------|
| MMK Net | **0.6685** | **0.6228** | **0.5962** | **0.5202** |
| ConvTokSWLA | 0.5803 | 0.5227 | 0.5780 | 0.5184 |
| InceptionTime | 0.6160 | 0.5631 | 0.5544 | 0.4868 |
| ConvTokMHSA | 0.3371 | 0.2080 | 0.3791 | 0.2745 |

*Note: ConvTokMHSA performs poorly on FC (global attention introduces cross-stage noise), validating the need for local architectures.*

### 3. End-to-End Diagnosis
Unified output (Healthy + N fault classes). Evaluated with MCWPM (Multi-Class Weighted Penalty Metric, α=2.5, β=1.0) emphasizing safety over false alarms.

```bash
# End-to-end baseline
python exp_diagnosis.py

# Two-stage LMSD (ConvTokMHSA + MMK Net)
python exp_diagnosis.py --model LMSD
```

**Key Results** (from paper Table 5):

| Model | Subset ACC | Subset F1 | Subset WF1 | Subset MCWPM | Overall ACC | Overall F1 | Overall WF1 | Overall MCWPM |
|-------|-----------|-----------|------------|--------------|-------------|------------|-------------|---------------|
| **LMSD** | **0.6656** | **0.4951** | **0.6634** | **0.6543** | 0.6291 | **0.4091** | **0.6300** | **0.6148** |
| ConvTokSWLA | 0.6568 | 0.4446 | 0.6475 | 0.6116 | **0.6424** | 0.3837 | 0.6299 | 0.5712 |
| InceptionTime | 0.6356 | 0.4557 | 0.6365 | 0.6338 | 0.5843 | 0.3570 | 0.5867 | 0.5652 |
| MMK Net | 0.6517 | 0.4745 | 0.6423 | 0.5903 | 0.6111 | 0.3188 | 0.6008 | 0.5417 |

**Training Efficiency**:
- LMSD: TTT=2001.63s (Overall), MSize=12.97MB
- InceptionTimeAttn: TTT=8388.47s, MSize=24.14MB

### 4. Interpretability (KEL)
Knowledge distillation-based temporal keyness extraction for physical interpretability.

```bash
python exp_kel_ad.py
# Teacher: ConvTokMHSA (e_layers=2)
# Student: Shallow variant (e_layers=1) with keyness output
```

Generates temporal keyness vectors indicating critical time steps for model decisions.

## Environment

**Tested Configuration**:
- Python 3.12, PyTorch 2.7.1+cu126
- CPU: Intel i7-13620H (32GB RAM)
- GPU: NVIDIA RTX 4070 (28GB VRAM)

*Note: Efficiency metrics (ET, TTT, IT32) reported above are specific to this hardware baseline.*

## Repository Structure

```
├── Datasets/               # Raw NGAFID data (not included, download via Setup)
├── ProcessedData/          # Preprocessed tensors (generated)
├── Models/                 # Architecture implementations
├── configs/                # Experiment configurations
├── tools/                  # Logging utilities
├── EXP_Logs/               # Experiment outputs (auto-generated)
├── prepare_diagnosis_data.py
├── exp_ad_detection.py
├── exp_fc_classification.py
├── exp_diagnosis.py
└── exp_kel_ad.py
```

## Implementation Notes

- **Data Leakage Prevention**: Z-score normalization parameters computed on training folds only; test fold data physically isolated at file level
- **Randomness Control**: 5-fold splits use fixed seed (350234); internal training uses variable seeds across 3 rounds to assess initialization sensitivity
- **Class Imbalance**: FC/Diagnosis tasks use proportional replication (Eq. 10 in paper) capped at majority class size to prevent distribution bias
- **Decoupled Training**: LMSD trains Health Analyzer (ConvTokMHSA) and Fault Analyzer (MMK Net) independently with gradient isolation, then cascades with frozen parameters

## Citation

*To be added after publication. This repository contains the official implementation of the proposed LMSD architecture.*

## License

Research code for academic use. Dataset usage follows NGAFID/Zenodo terms.
