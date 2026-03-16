#!/usr/bin/env python3
"""
KEL Configuration for Fault Classification (FC)
================================================
Knowledge Distillation with Keyness Extraction for multi-class fault classification.

Usage:
    from configs.kd_fc_config import get_args, get_args_meanings
    args, setting = get_args()
"""

import sys
from pathlib import Path
from typing import Tuple, Dict

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from tools.kel_model_factory import KELStudentFactory, KELTeacherFactory


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_args_meanings() -> Dict[str, str]:
    """Parameter documentation."""
    return {
        # Hardware
        "gpu": "GPU device ID.",
        "use_gpu": "Enable CUDA acceleration.",
        "use_multi_gpu": "Enable multi-GPU training.",
        "devices": "Comma-separated GPU IDs.",

        # Distillation
        "temperature": "Softmax temperature for distillation.",
        "scale": "Keyness vector scaling factor.",

        # Training
        "learning_rate": "Initial learning rate.",
        "batch_size": "Mini-batch size.",
        "train_epochs": "Maximum training epochs.",
        "patience": "Early stopping patience.",
        "lradj": "LR adjustment strategy.",

        # Data
        "data_select_pattern": "Dataset variant: '2days' or 'full'.",
        "da_this_exp": "Enable data augmentation for minority classes.",
        "da_method": "DA method: 'random_copy' or 'random_noise'.",
        "num_copies": "Number of copies for random_copy augmentation.",

        # Teacher (Frozen)
        "TchrArgs.model_name": "Teacher architecture.",
        "TchrArgs.teacher_path": "Path to pretrained teacher checkpoint.",
        "TchrArgs.inception_layers": "Teacher depth (InceptionTime/MMK Net).",
        "TchrArgs.filters": "Teacher convolution filters.",

        # Student (Trainable)
        "StdntArgs.model_name": "Student architecture (KeynessV2).",
        "StdntArgs.is_only_time": "Temporal keyness extraction mode.",
        "StdntArgs.inception_layers": "Student depth (shallower than teacher).",
        "StdntArgs.clasnum": "Number of fault classes (19 for 2days, 36 for full).",
    }


def get_args() -> Tuple[dotdict, str]:
    """Construct KEL-FC configuration."""
    args = dotdict()

    # Hardware
    args.gpu = 0
    args.devices = '0'
    args.use_gpu = True
    args.use_multi_gpu = False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # Task
    args.data = 'NGAFID'
    args.data_select_pattern = "2days"  # or "full"
    args.model_name = "MMK_Net_KD"  # Experiment identifier

    # Distillation params
    args.temperature = 1.2
    args.scale = 32

    # Training
    args.learning_rate = 1e-4
    args.batch_size = 32
    args.train_epochs = 100
    args.patience = 3
    args.lradj = 'type3'
    args.testfoldid = 2
    args.save_model_path = None

    # Data Augmentation (FC specific - critical for class imbalance)
    args.da_this_exp = True  # Enable DA for minority fault classes
    args.da_method = "random_copy"
    args.num_copies = 2

    # ==================== Teacher Configuration (Frozen) ====================
    args.TchrArgs = dotdict()
    args.TchrArgs.model_name = "MMK_Net"
    args.TchrArgs.data = 'NGAFID'
    args.TchrArgs.is_only_time = True

    # Use standard factory for teacher (non-keyness)
    tchr_factory = KELTeacherFactory()
    args.TchrArgs.model = tchr_factory.get_model_class(args.TchrArgs.model_name)

    # Teacher architecture (deeper)
    args.TchrArgs.inception_layers = 4
    args.TchrArgs.clasnum = 19  # 2days: 19 faults, full: 36 faults
    args.TchrArgs.dropout = 0.01
    args.TchrArgs.in_dim = 23
    args.TchrArgs.full_len = 2048
    args.TchrArgs.hidden_dim = 2048
    args.TchrArgs.filters = 256
    args.TchrArgs.use_amp = False
    args.TchrArgs.activation = 'gelu'

    # Pretrained teacher path (MUST be trained FC model)
    args.TchrArgs.teacher_path = (
        f"{PROJECT_ROOT}/ModelCheckpoints/FC/MMK_Net/FC_2days/FC_MMK_Net_2days_il4_ft256_cn19_fold2/checkpoint.pth"
    )

    # ==================== Student Configuration (Trainable) ====================
    args.StdntArgs = dotdict()
    args.StdntArgs.model_name = "MMK_Net"
    args.StdntArgs.data = 'NGAFID'
    args.StdntArgs.is_only_time = True  # KeynessV2 variant

    # Use KEL factory for student (KeynessV2)
    stdnt_factory = KELStudentFactory()
    args.StdntArgs.model = stdnt_factory.get_model_class(
        args.StdntArgs.model_name,
        args.StdntArgs.is_only_time
    )

    # Student architecture (shallower for efficiency)
    args.StdntArgs.inception_layers = 3  # Shallower than teacher (4)
    args.StdntArgs.clasnum = 19
    args.StdntArgs.dropout = 0.01
    args.StdntArgs.in_dim = 23
    args.StdntArgs.full_len = 2048
    args.StdntArgs.hidden_dim = 2048
    args.StdntArgs.filters = 256
    args.StdntArgs.use_amp = False
    args.StdntArgs.activation = 'gelu'
    args.StdntArgs.scale = args.scale  # Keyness scaling

    # ==================== Checkpoint Paths ====================
    args.checkpoints = f'./KD/{args.model_name}'

    # ==================== Setting String ====================
    setting = 'KELFC_{}_{}_Tchr{}x{}_Stdnt{}x{}_sc{}_t{}_fold{}'.format(
        args.model_name,
        args.data_select_pattern,
        args.TchrArgs.inception_layers,
        args.TchrArgs.filters,
        args.StdntArgs.inception_layers,
        args.StdntArgs.filters,
        args.scale,
        args.temperature,
        args.testfoldid
    )

    return args, setting