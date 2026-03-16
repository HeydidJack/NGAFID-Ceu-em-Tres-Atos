#!/usr/bin/env python3
"""
Fault Classification (FC) Configuration
=======================================
Configuration template for multi-class fault classification on NGAFID dataset.
Trains only on anomalous samples (filters out healthy flights).

Usage:
    from configs.fc_config import get_args, get_args_meanings

    args, setting = get_args()
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
        "data_select_pattern": "Dataset variant: '2days' (19 classes) or 'full' (36 classes).",
        "in_dim": "Number of input sensor channels (23).",
        "full_len": "Temporal sequence length (2048 timestamps).",
        "clasnum": "Number of fault classes (auto-set: 19 for 2days, 36 for full).",
        "health_index": "Index of healthy class in one-hot labels (0), used for filtering.",

        # Model Architecture (MMK_Net / InceptionTime family)
        "model_name": "Model identifier. MMK_Net recommended for FC (local features).",
        "model": "Model class (auto-populated).",
        "inception_layers": "Depth of Inception/MMK blocks (equivalent to num_layers). Default: 4.",
        "num_layers": "Alias for inception_layers for compatibility.",
        "filters": "Convolution output channels. Default: 256 (MMK_Net).",
        "hidden_dim": "Final projection dimension. Default: 2048.",
        "dropout": "Dropout ratio. Default: 0.01.",

        # ConvTok-specific (if using attention-based models)
        "L_patch": "Patch length for tokenization (ConvTok series only).",
        "token_dim": "Token embedding dimension.",
        "e_layers": "Transformer encoder depth (ConvTok series).",
        "n_heads": "Attention heads (ConvTok series).",
        "d_ff": "Feed-forward dimension (ConvTok series).",
        "viewindow": "Sliding window half-width (ConvTokSWLA only).",

        # Training
        "learning_rate": "Initial learning rate (1e-4 for MMK_Net).",
        "batch_size": "Mini-batch size (32).",
        "patience": "Early stopping patience (epochs).",
        "train_epochs": "Maximum training epochs.",
        "lradj": "LR adjustment: 'type1', 'type2', or 'type3' (halve on plateau).",
        "activation": "Non-linearity: 'gelu' or 'relu'.",
        "use_amp": "Automatic Mixed Precision flag.",

        # Data Augmentation (for handling class imbalance in FC)
        "da_this_exp": "Enable data augmentation for minority fault classes.",
        "augmentation_method": "Method: 'random_copy' or 'random_noise'.",
        "num_copies": "Number of augmented copies per minority sample.",

        # Task-specific
        "testfoldid": "Which fold (0-4) to use as test set.",
        "checkpoints": "Directory for model checkpoints.",
        "save_model_path": "Optional override for checkpoint path.",
    }


def get_args() -> Tuple[dotdict, str]:
    """Construct argument namespace for Fault Classification."""
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
    # MMK_Net: Multi-scale Micro-Kernel Network (restricted receptive field)
    # Optimal for FC task requiring fine-grained local feature extraction
    args.model_name = "MMK_Net"
    args.data = 'NGAFID'

    factory = ModelFactory()
    args.model = factory.get_model_class(args.model_name)
    args.checkpoints = f'./FC/{args.model_name}'
    args.save_model_path = None

    # ==================== Data Configuration ====================
    args.data_select_pattern = "2days"  # "2days" (19 classes) or "full" (36 classes)
    args.in_dim = 23
    args.full_len = 2048
    # Auto-set class number based on dataset variant
    args.clasnum = 19 if args.data_select_pattern == "2days" else 36
    args.health_index = 0  # Index of healthy class (filtered out in FC)

    # ==================== Architecture Parameters ====================
    # MMK_Net specific (InceptionTime family with restricted receptive fields)
    args.inception_layers = 4  # Depth
    args.num_layers = 4  # Alias for compatibility
    args.filters = 256  # Convolution channels (128 for smaller datasets)
    args.hidden_dim = 2048  # Projection dimension
    args.dropout = 0.01

    # Unused in MMK_Net but kept for unified config interface
    args.L_patch = 4
    args.token_dim = 512
    args.e_layers = 4
    args.n_heads = 4
    args.d_ff = 1024
    args.viewindow = 4

    # ==================== Training Configuration ====================
    args.learning_rate = 1e-4  # MMK_Net uses 1e-4 (same as ConvTok)
    args.batch_size = 32
    args.train_epochs = 100
    args.patience = 3
    args.lradj = 'type3'
    args.activation = 'gelu'
    args.use_amp = False
    args.testfoldid = 2

    # ==================== Data Augmentation ====================
    # Critical for FC due to severe class imbalance (some faults have <10 samples)
    args.da_this_exp = True  # 是否启用 DA
    args.da_method = "random_copy"  # 方法名：random_copy, random_noise, jitter, time_warp 等
    args.num_copies = 3

    # ==================== Setting String ====================
    setting = 'FC_{}_{}_il{}_ft{}_cn{}_fold{}'.format(
        args.model_name,
        args.data_select_pattern,
        args.inception_layers,
        args.filters,
        args.clasnum,
        args.testfoldid
    )

    return args, setting


if __name__ == "__main__":
    args, setting = get_args()
    print(f"FC Configuration loaded: {setting}")
    print(f"Classes: {args.clasnum} ( {'2days' if args.clasnum == 19 else 'full'} )")
    print(f"Model: {args.model_name}")