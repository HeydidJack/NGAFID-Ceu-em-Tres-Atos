#!/usr/bin/env python3
"""
Knowledge Distillation Configuration
====================================
Configuration for KEL (Knowledge Distillation-based Explainability Learning).
Teacher-Student architecture with temporal keyness extraction.

Usage:
    from configs.kd_config import get_args, get_args_meanings

    args, setting = get_args()
    # Access teacher config: args.TchrArgs.xxx
    # Access student config: args.StdntArgs.xxx
    # Access distillation params: args.temperature, args.scale
"""

import sys
from pathlib import Path
from typing import Tuple, Dict

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from tools.kel_model_factory import KELStudentFactory, KELTeacherFactory


class dotdict(dict):
    """Dictionary with attribute-style access."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_args_meanings() -> Dict[str, str]:
    """Parameter documentation."""
    return {
        # Hardware
        "gpu": "GPU device ID for training.",
        "use_gpu": "Enable CUDA acceleration.",
        "use_multi_gpu": "Enable multi-GPU training.",
        "devices": "Comma-separated GPU IDs.",

        # Distillation Parameters
        "temperature": "Softmax temperature for distillation (T). Higher = softer distributions.",
        "scale": "Keyness vector scaling factor for distillation loss weighting.",
        "alpha": "Weight for hard target loss (1-alpha for soft distillation loss).",

        # Training
        "learning_rate": "Initial learning rate for student.",
        "batch_size": "Mini-batch size.",
        "train_epochs": "Maximum training epochs.",
        "patience": "Early stopping patience.",
        "lradj": "Learning rate adjustment strategy.",

        # Teacher Model (Frozen)
        "TchrArgs.model_name": "Teacher architecture (standard model).",
        "TchrArgs.teacher_path": "Path to pretrained teacher checkpoint.",
        "TchrArgs.e_layers": "Teacher encoder layers (deeper than student).",
        "TchrArgs.d_ff": "Teacher feed-forward dimension.",

        # Student Model (Trainable)
        "StdntArgs.model_name": "Student architecture (KeynessV2 variant).",
        "StdntArgs.is_only_time": "If True, student outputs temporal keyness only.",
        "StdntArgs.e_layers": "Student encoder layers (shallower than teacher).",
        "StdntArgs.scale": "Keyness extraction scaling factor.",
    }


def get_args() -> Tuple[dotdict, str]:
    """
    Construct KEL training configuration.

    Architecture:
    - Teacher: Deep standard model (e.g., ConvTokMHSA e_layers=2)
    - Student: Shallow KeynessV2 model (e.g., TimeKeynessConvTokMHSA e_layers=1)
    - Distillation: Student learns from teacher's soft labels + temporal keyness alignment
    """
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

    # ==================== Task Configuration ====================
    args.data = 'NGAFID'
    args.data_select_pattern = "2days"  # or "full"
    args.model_name = "ConvTokMHSA_KD"  # Identifier for this KD experiment

    # ==================== Distillation Parameters ====================
    args.temperature = 1.2  # Distillation temperature T
    args.scale = 4  # Keyness scaling factor (StdntArgs.scale)
    args.alpha = 0.3  # Hard target weight (0.3 hard + 0.7 soft)
    args.distill_loss_weight = 1.0  # Weight for KL divergence loss

    # ==================== Training Hyperparameters ====================
    args.learning_rate = 1e-4
    args.batch_size = 32
    args.train_epochs = 50
    args.patience = 3
    args.lradj = 'type3'  # Halve LR on plateau
    args.testfoldid = 2  # Which fold to test on

    # ==================== Teacher Configuration (Frozen) ====================
    args.TchrArgs = dotdict()
    args.TchrArgs.model_name = "ConvTokMHSA"
    args.TchrArgs.data = 'NGAFID'

    # Teacher uses standard factory (non-keyness)
    tchr_factory = KELTeacherFactory()
    args.TchrArgs.model = tchr_factory.get_model_class(args.TchrArgs.model_name)

    # Teacher paths and hyperparameters (deeper network)
    args.TchrArgs.L_patch = 4
    args.TchrArgs.clasnum = 2
    args.TchrArgs.full_len = 2048
    args.TchrArgs.e_layers = 2  # Teacher: deeper (2 layers)
    args.TchrArgs.dropout = 0.01
    args.TchrArgs.n_heads = 4
    args.TchrArgs.in_dim = 23
    args.TchrArgs.d_ff = 512
    args.TchrArgs.token_dim = 128
    args.TchrArgs.factor = 5
    args.TchrArgs.output_attention = False
    args.TchrArgs.use_amp = False
    args.TchrArgs.activation = 'gelu'

    # Pretrained teacher checkpoint path (MUST exist)
    args.TchrArgs.teacher_path = (
        f"{PROJECT_ROOT}/ModelCheckpoints/AD/ConvTokMHSA/AD_ConvTokMHSA_2days_lp4_td128_nh4_el2_df512_fold0/checkpoint.pth"
    )

    # ==================== Student Configuration (Trainable) ====================
    args.StdntArgs = dotdict()
    args.StdntArgs.model_name = "ConvTokMHSA"  # Will map to KeynessV2 variant
    args.StdntArgs.data = 'NGAFID'
    args.StdntArgs.is_only_time = True  # Temporal keyness extraction

    # Student uses KEL factory (KeynessV2 variants)
    stdnt_factory = KELStudentFactory()
    args.StdntArgs.model = stdnt_factory.get_model_class(
        args.StdntArgs.model_name,
        args.StdntArgs.is_only_time
    )

    # Student hyperparameters (shallower network for efficiency)
    args.StdntArgs.L_patch = 4
    args.StdntArgs.clasnum = 2
    args.StdntArgs.full_len = 2048
    args.StdntArgs.e_layers = 1  # Student: shallower (1 layer)
    args.StdntArgs.dropout = 0.01
    args.StdntArgs.n_heads = 4
    args.StdntArgs.in_dim = 23
    args.StdntArgs.d_ff = 512
    args.StdntArgs.token_dim = 128
    args.StdntArgs.factor = 5
    args.StdntArgs.output_attention = False
    args.StdntArgs.use_amp = False
    args.StdntArgs.activation = 'gelu'
    args.StdntArgs.scale = args.scale  # Keyness scaling factor

    # ==================== Checkpoint Paths ====================
    args.checkpoints = f'./KD/{args.model_name}'
    args.save_model_path = None  # Override for specific experiments

    # ==================== Experiment Setting String ====================
    setting = (
        'KEL_{}_{}_Tchr{}x{}_Stdnt{}x{}_sc{}_t{}_lp{}_td{}_nh{}_fold{}'
    ).format(
        args.model_name,
        args.data_select_pattern,
        args.TchrArgs.e_layers,  # Teacher depth
        args.TchrArgs.d_ff,
        args.StdntArgs.e_layers,  # Student depth
        args.StdntArgs.d_ff,
        args.scale,
        args.temperature,
        args.StdntArgs.L_patch,
        args.StdntArgs.token_dim,
        args.StdntArgs.n_heads,
        args.testfoldid
    )

    return args, setting


def print_config_help():
    """Print parameter documentation."""
    meanings = get_args_meanings()
    print("=" * 70)
    print("KEL (Knowledge Distillation) CONFIGURATION REFERENCE")
    print("=" * 70)
    for param, desc in meanings.items():
        print(f"{param:25s}: {desc}")
    print("=" * 70)


if __name__ == "__main__":
    args, setting = get_args()
    print(f"KEL Configuration loaded.")
    print(f"Experiment setting: {setting}")
    print(f"Teacher: {args.TchrArgs.model_name} (layers={args.TchrArgs.e_layers})")
    print(
        f"Student: {args.StdntArgs.model_name} (layers={args.StdntArgs.e_layers}, is_only_time={args.StdntArgs.is_only_time})")
    print(f"Distillation: T={args.temperature}, scale={args.scale}")
    print_config_help()