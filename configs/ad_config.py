#!/usr/bin/env python3
"""
Anomaly Detection Configuration
===============================
Configuration template for Anomaly Detection (AD) task on NGAFID dataset.
All hyperparameters are model-agnostic; switch models by changing `model_name`.

Usage:
    from configs.ad_config import get_args, get_args_meanings, print_config_help

    args, setting = get_args()
    # Access via dot notation: args.model_name, args.learning_rate, etc.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any

# Add project root to path for importing tools
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from tools.model_factory import ModelFactory


class dotdict(dict):
    """
    Dictionary subclass allowing attribute-style access (e.g., args.learning_rate).
    Supports dict.get, dict.__setitem__, dict.__delitem__ semantics.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_args_meanings() -> Dict[str, str]:
    """
    Returns dictionary mapping parameter names to their English descriptions.
    Useful for generating documentation and debug logs.
    """
    return {
        # Hardware Configuration
        "gpu": "GPU device ID (int). Set to 0 for single GPU, -1 for CPU.",
        "devices": "Comma-separated GPU IDs for multi-GPU training (e.g., '0,1,2').",
        "use_gpu": "Boolean flag to enable CUDA acceleration.",
        "use_multi_gpu": "Boolean flag to enable DataParallel across multiple GPUs.",

        # Data Configuration
        "data": "Dataset identifier (e.g., 'NGAFID').",
        "data_select_pattern": "Dataset variant: '2days' (subset) or 'full' (complete).",
        "in_dim": "Number of input sensor channels (23 for NGAFID standard).",
        "full_len": "Temporal sequence length after resampling (2048 timestamps).",
        "clasnum": "Number of output classes (2 for binary AD: Healthy vs Anomaly).",

        # Model Architecture Selection
        "model_name": "Model architecture identifier. Must match ModelFactory registry.",
        "model": "Actual model class (auto-populated by ModelFactory).",

        # Generic Architecture Parameters (interpretation varies by model)
        "num_layers": "Stacked layer depth (for LSTM/InceptionTime/MMK Net).",
        "encoder_layers": "Transformer encoder depth (for ConvTok series).",
        "hidden_dim": "Hidden representation dimension (model internal).",
        "filters": "CNN convolution output channels (for InceptionTime/MMK Net).",
        "n_heads": "Number of parallel attention heads (for attention-based models).",
        "d_ff": "Feed-forward network hidden dimension (for Transformer-based models).",
        "token_dim": "Token embedding dimension (for ConvTok series).",
        "L_patch": "Patch length for temporal tokenization (ConvTok series only).",
        "viewindow": "Sliding window half-width (ConvTokSWLA only).",

        # Training Hyperparameters
        "learning_rate": "Initial learning rate (1e-4 for most models, 3e-5 for MMK Net).",
        "batch_size": "Mini-batch size (32 for deep models, larger for shallow baselines).",
        "dropout": "Dropout regularization ratio (0.01 for ConvTok/InceptionTime).",
        "patience": "Early stopping patience (epochs without improvement).",
        "train_epochs": "Maximum training epochs (safety limit).",
        "lradj": "Learning rate adjustment strategy: 'type1', 'type2', or 'type3' (halve on plateau).",
        "activation": "Non-linearity: 'gelu' (recommended) or 'relu'.",
        "use_amp": "Automatic Mixed Precision training flag (not recommended for stability).",

        # Task-Specific
        "testfoldid": "Which fold (0-4) to use as test set for cross-validation.",
        "output_attention": "Flag to return attention weights (for interpretability).",
        "checkpoints": "Directory path for model checkpoint storage.",
        "save_model_path": "Optional override for custom checkpoint path.",
    }


def get_args() -> Tuple[dotdict, str]:
    """
    Constructs argument namespace and generates experiment setting string.

    Returns:
        Tuple of (args_namespace, setting_string)
        - args_namespace: dotdict containing all hyperparameters
        - setting_string: Formatted experiment identifier for logging/folder naming
    """
    args = dotdict()

    # ==================== Hardware Configuration ====================
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
    # Simply change model_name to switch architectures.
    # Available: Bi-LSTM, CNN, MLP, InceptionTime, MMK_Net,
    #            ConvTokMHSA, ConvTokSWLA, ConvTokMWLA, ConvTokLPLA, LMSD
    args.model_name = "ConvTokMHSA"
    args.data = 'NGAFID'

    # Auto-resolve model class from factory (no manual import needed)
    factory = ModelFactory()
    args.model = factory.get_model_class(args.model_name)

    # Checkpoint path (auto-generated based on model name)
    args.checkpoints = f'./AD/{args.model_name}'
    args.save_model_path = None  # Override if needed for specific experiments

    # ==================== Data Configuration ====================
    args.data_select_pattern = "2days"  # or "full"
    args.in_dim = 23
    args.full_len = 2048
    args.clasnum = 2  # Binary: Healthy (0) vs Anomaly (1)

    # ==================== Architecture Parameters ====================
    # Note: Not all parameters are used by all models.
    # Unused params are ignored by models that don't require them.

    # ConvTok Series Parameters
    args.L_patch = 4
    args.token_dim = 128
    args.encoder_layers = 2  # e_layers in some legacy code
    args.e_layers = 2  # Alias for compatibility
    args.n_heads = 4
    args.d_ff = 512

    # InceptionTime/MMK Net Parameters
    args.num_layers = 4
    args.inception_layers = 4  # Alias for compatibility
    args.filters = 256
    args.hidden_dim = 2048

    # Bi-LSTM Parameters
    # args.num_layers = 3  # Override above if using Bi-LSTM
    # args.hidden_dim = 128

    # ConvTokSWLA/MWLA Specific
    args.viewindow = 4

    # ==================== Training Configuration ====================
    args.dropout = 0.01
    args.activation = 'gelu'
    args.output_attention = False
    args.use_amp = False

    # Optimization
    args.learning_rate = 1e-4  # Use 3e-5 for MMK_Net, 1e-3 for Bi-LSTM
    args.batch_size = 32
    args.train_epochs = 100
    args.patience = 3
    args.lradj = 'type3'  # Halve LR when loss plateaus

    # Cross-validation
    args.testfoldid = 0  # Test on fold 0, train on 1,2,3,4

    # ==================== Experiment Setting String ====================
    # Format: {model}_{data}_lp{patch}_td{dim}_nh{heads}_el{layers}_df{ffdim}_fold{testfold}
    setting = 'AD_{}_{}_lp{}_td{}_nh{}_el{}_df{}_fold{}'.format(
        args.model_name,
        args.data_select_pattern,
        args.L_patch,
        args.token_dim,
        args.n_heads,
        args.e_layers,
        args.d_ff,
        args.testfoldid
    )

    return args, setting


def print_config_help():
    """Prints formatted parameter documentation."""
    meanings = get_args_meanings()
    print("=" * 70)
    print("ANOMALY DETECTION CONFIGURATION REFERENCE")
    print("=" * 70)
    for param, desc in meanings.items():
        print(f"{param:20s}: {desc}")
    print("=" * 70)


if __name__ == "__main__":
    # Test configuration loading
    args, setting = get_args()
    print(f"Configuration loaded successfully.")
    print(f"Experiment setting: {setting}")
    print(f"Model selected: {args.model_name} -> {args.model.__name__}")
    print_config_help()