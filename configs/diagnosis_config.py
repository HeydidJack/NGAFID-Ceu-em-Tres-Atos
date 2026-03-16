#!/usr/bin/env python3
"""
End-to-End Diagnosis Configuration
==================================
Configuration for unified health diagnosis (Anomaly Detection + Fault Classification).
Evaluates on full label space with MCWPM (Multi-Class Weighted Penalty Metric).

Usage:
    from configs.diagnosis_config import get_args, get_args_meanings

    args, setting = get_args()
    # Diagnosis uses full labels: 20 classes (2days) or 37 classes (full)
"""

import sys
from pathlib import Path
from typing import Tuple, Dict

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from tools.model_factory import ModelFactory


class dotdict(dict):
    """Dictionary subclass allowing attribute-style access."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_args_meanings() -> Dict[str, str]:
    """Returns parameter documentation."""
    return {
        # Hardware
        "gpu": "GPU device ID (int). Set to 0 for single GPU, -1 for CPU.",
        "devices": "Comma-separated GPU IDs for multi-GPU training.",
        "use_gpu": "Boolean flag to enable CUDA acceleration.",
        "use_multi_gpu": "Boolean flag to enable DataParallel.",

        # Data
        "data": "Dataset identifier (e.g., 'NGAFID').",
        "data_select_pattern": "Dataset variant: '2days' (20 classes) or 'full' (37 classes).",
        "in_dim": "Number of input sensor channels (23).",
        "full_len": "Temporal sequence length (2048 timestamps).",
        "clasnum": "Total classes including healthy: 20 (2days) or 37 (full).",
        "health_index": "Index of healthy class (0), used for MCWPM calculation.",

        # Model Architecture (ConvTokMHSA for Diagnosis - deep config)
        "model_name": "Model identifier. ConvTokMHSA for global context modeling.",
        "model": "Model class (auto-populated).",
        "L_patch": "Patch length for tokenization. Default: 4.",
        "token_dim": "Token embedding dimension. Default: 512 (Diagnosis uses deep config).",
        "e_layers": "Transformer encoder depth. Default: 4 (Diagnosis task requires deep layers).",
        "n_heads": "Number of attention heads. Default: 4.",
        "d_ff": "Feed-forward dimension. Default: 1024.",
        "dropout": "Dropout ratio. Default: 0.01.",

        # ConvTok variants specific
        "viewindow": "Sliding window half-width (only for SWLA/MWLA variants).",

        # Training
        "learning_rate": "Initial learning rate. Default: 1e-4.",
        "batch_size": "Mini-batch size (32).",
        "patience": "Early stopping patience (epochs).",
        "train_epochs": "Maximum training epochs.",
        "lradj": "LR adjustment: 'type1', 'type2', or 'type3' (halve on plateau).",
        "activation": "Non-linearity: 'gelu' or 'relu'.",
        "use_amp": "Automatic Mixed Precision flag.",

        # Data Augmentation (for handling imbalance in end-to-end setting)
        "da_this_exp": "Enable data augmentation for minority classes.",
        "augmentation_method": "Method: 'random_copy' or 'random_noise'.",
        "num_copies": "Number of augmented copies per minority sample.",

        # Task-specific
        "testfoldid": "Which fold (0-4) to use as test set.",
        "checkpoints": "Directory for model checkpoints.",
        "save_model_path": "Optional override for checkpoint path.",

        # MCWPM Parameters (Diagnosis specific)
        "mcwpm_alpha": "Penalty factor for missed detection (False Negative). Default: 2.5.",
        "mcwpm_beta": "Penalty factor for false alarm (False Positive). Default: 1.0.",
    }


def get_args() -> Tuple[dotdict, str]:
    """Construct argument namespace for End-to-End Diagnosis."""
    args = dotdict()

    # ==================== Hardware ====================
    args.gpu = 0
    args.devices = '0'
    args.use_gpu = True
    args.use_multi_gpu = False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # ==================== Model Selection ====================
    # ConvTokMHSA with deep configuration for end-to-end diagnosis
    args.model_name = "ConvTokMHSA"
    args.data = 'NGAFID'

    factory = ModelFactory()
    args.model = factory.get_model_class(args.model_name)
    args.checkpoints = f'./Diagnosis/{args.model_name}'
    args.save_model_path = None

    # ==================== Data Configuration ====================
    # Diagnosis uses full label space: Healthy + All Fault Types
    args.data_select_pattern = "2days"  # "2days" (20 classes: 1H+19F) or "full" (37 classes: 1H+36F)
    args.in_dim = 23
    args.full_len = 2048
    # Auto-calculate: 1 (healthy) + num_faults (19 or 36)
    args.clasnum = 20 if args.data_select_pattern == "2days" else 37
    args.health_index = 0  # Index 0 is healthy class (for MCWPM)

    # ==================== Architecture Parameters ====================
    # ConvTokMHSA deep configuration (as per ModelConfigsExplanation.md for FC/Diagnosis)
    args.L_patch = 4
    args.token_dim = 512  # Increased from 128 (AD) to 512 (Diagnosis)
    args.e_layers = 4  # Increased from 2 (AD) to 4 (Diagnosis)
    args.n_heads = 4
    args.d_ff = 1024  # Increased from 512 (AD) to 1024 (Diagnosis)
    args.dropout = 0.01
    args.viewindow = 4  # Placeholder for SWLA variants

    # Unused placeholders for compatibility with InceptionTime family
    args.inception_layers = 4
    args.num_layers = 4
    args.filters = 256
    args.hidden_dim = 2048

    # ==================== Training Configuration ====================
    args.learning_rate = 1e-4
    args.batch_size = 32
    args.train_epochs = 100
    args.patience = 3
    args.lradj = 'type3'
    args.activation = 'gelu'
    args.use_amp = False
    args.testfoldid = 0

    # ==================== Data Augmentation ====================
    # Optional: Address imbalance between healthy (majority) and faults (minority)
    args.da_this_exp = True  # 是否启用 DA
    args.da_method = "random_copy"  # 方法名：random_copy, random_noise, jitter, time_warp 等
    args.num_copies = 3

    # ==================== MCWPM Configuration ====================
    args.mcwpm_alpha = 2.5  # Penalty for missed anomaly (safety-critical)
    args.mcwpm_beta = 1.0  # Penalty for false alarm (cost-sensitive)

    # ==================== Setting String ====================
    setting = 'Diagnosis_{}_{}_lp{}_td{}_nh{}_el{}_df{}_cn{}_fold{}'.format(
        args.model_name,
        args.data_select_pattern,
        args.L_patch,
        args.token_dim,
        args.n_heads,
        args.e_layers,
        args.d_ff,
        args.clasnum,
        args.testfoldid
    )

    return args, setting


if __name__ == "__main__":
    args, setting = get_args()
    print(f"Diagnosis Configuration loaded: {setting}")
    print(f"Total Classes: {args.clasnum} (1 Healthy + {args.clasnum - 1} Faults)")
    print(f"Model: {args.model_name} (Deep Config: e_layers={args.e_layers}, token_dim={args.token_dim})")