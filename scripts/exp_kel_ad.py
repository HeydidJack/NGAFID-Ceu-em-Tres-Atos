#!/usr/bin/env python3
"""
KEL (Knowledge Distillation-based Explainability Learning) for AD
==================================================================
Distills knowledge from deep teacher (ConvTokMHSA) to shallow student (KEL_ConvTokMHSA)
while extracting temporal keyness vectors for physical interpretability.

Teacher: Deep standard model (frozen, pretrained)
Student: Shallow KEL model (trainable, outputs keyness vector)
"""

import os
import sys
import time
import pickle
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from tools.exp_logger import init_exp_log, print_aptxt, append_text_to_file, get_divider, get_current_datetime_string
from configs.ad_kd_config import get_args, get_args_meanings

# Global log paths (legacy compatibility)
Exp_log_path = None
Exp_outcome_path = None


def check_and_generate_data(output_folder=f"{PROJECT_ROOT}\ProcessedData\ProcessedData_Diagnosis_DeD_2days"):
    """Verify processed data exists; terminate if missing."""
    if not os.path.exists(output_folder):
        print(f"[FATAL] Processed data directory not found: {output_folder}")
        print("\nPlease generate the dataset first using:")
        dataset_type = "2days" if "2days" in output_folder else "full"
        print(f"  python scripts/prepare_diagnosis_data.py --dataset {dataset_type}")
        sys.exit(1)

    pkl_files = [f for f in os.listdir(output_folder) if f.endswith(".pkl")]
    if len(pkl_files) == 0:
        print(f"[FATAL] Directory exists but no .pkl files found in: {output_folder}")
        print("\nPlease regenerate the dataset:")
        dataset_type = "2days" if "2days" in output_folder else "full"
        print(f"  python scripts/prepare_diagnosis_data.py --dataset {dataset_type}")
        sys.exit(1)

    print(f"[OK] Found {len(pkl_files)} processed data files in {output_folder}")


def LoadDataFolds(data_select_pattern="2days"):
    """Load 5-fold CV data for binary AD (Healthy vs Anomaly)."""
    fold_data_list = []
    fold_label_list = []
    fold_index_list = []

    for fold_id in range(5):
        pkl_path = f"{PROJECT_ROOT}\ProcessedData\ProcessedData_Diagnosis_DeD_{data_select_pattern}/diagnosis_fold{fold_id}_2048.pkl"
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(pkl_path)

        with open(pkl_path, 'rb') as f:
            fold_dict = pickle.load(f)

        fold_data = np.array(fold_dict['train'], dtype=np.float32)
        fold_label = np.array(fold_dict['label'], dtype=np.float32)
        fold_indexes = fold_dict['index']

        health_index = 0
        new_label = np.zeros((len(fold_label), 2), dtype=np.float32)
        new_label[:, 0] = (fold_label[:, health_index] == 1).astype(np.float32)
        new_label[:, 1] = 1 - new_label[:, 0]

        fold_label_list.append(new_label)
        fold_data_list.append(fold_data)
        fold_index_list.append(fold_indexes)

        print_aptxt(
            f'Fold {fold_id}: data shape {fold_data.shape}, label shape {new_label.shape} loaded',
            Exp_log_path
        )

    print_aptxt("Data loading completed\n" + get_divider(), Exp_log_path)
    return fold_data_list, fold_label_list, fold_index_list


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.is_stop_descending = False

    def __call__(self, val_loss, model, path, is_save=True):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, is_save)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.is_stop_descending = True
            print_aptxt(f'EarlyStopping counter: {self.counter} out of {self.patience}', Exp_log_path)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, is_save)
            self.counter = 0
            self.is_stop_descending = False

    def save_checkpoint(self, val_loss, model, path, is_save=True):
        if self.verbose:
            print_aptxt(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...',
                        Exp_log_path)
        if is_save:
            os.makedirs(path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args, is_stop_descending=False, last_lr=5e-5):
    """Learning rate scheduler with type3 support (halve on plateau)."""
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {20: 5e-5, 40: 1e-5, 60: 5e-6, 80: 1e-6, 100: 5e-7}
    elif args.lradj == 'type3':
        if is_stop_descending and last_lr > 5e-12:
            lr = last_lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print_aptxt(f'Updating learning rate to {lr}', Exp_log_path)
            return lr
        return last_lr

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print_aptxt(f'Updating learning rate to {lr}', Exp_log_path)
    return last_lr


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(f'cuda:{self.args.gpu}')
            print_aptxt(f'Use GPU: cuda:{self.args.gpu}', Exp_log_path)
        else:
            device = torch.device('cpu')
            print_aptxt('Use CPU', Exp_log_path)
        return device

    def _build_model(self):
        raise NotImplementedError


class Exp_KEL(Exp_Basic):
    """KEL Training Experiment for AD Task."""

    def __init__(self, args):
        super(Exp_KEL, self).__init__(args)
        self.have_got_dataset = False
        self.scaler = None
        self.teacher_model = None
        self.model = None  # Student model

    def _check_model_arcsize(self):
        """Check student model size."""
        print_aptxt(get_divider() + "Student Model Size Information:", Exp_log_path)
        example_model = self.args.StdntArgs.model(self.args.StdntArgs)
        total_memory = sum(p.storage().nbytes() for p in example_model.parameters())
        print_aptxt(f"Total memory usage (bytes): {total_memory}", Exp_log_path)
        print_aptxt(f"Total memory usage (MB): {total_memory / (1024 * 1024):.2f}", Exp_log_path)
        torch.cuda.empty_cache()

    def _build_student(self):
        """Build student model (KEL variant)."""
        model = self.args.StdntArgs.model(self.args.StdntArgs).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _build_teacher(self):
        """Build teacher model (standard architecture) and load pretrained weights."""
        model = self.args.TchrArgs.model(self.args.TchrArgs).float()
        model.load_state_dict(torch.load(self.args.TchrArgs.teacher_path, map_location='cpu'))
        model.to(self.device)
        model.eval()
        return model

    def _init_model(self):
        """Initialize both teacher (frozen) and student (trainable)."""
        self.model = self._build_student().to(self.device)
        self.teacher_model = self._build_teacher()
        # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def _get_data(self, fold_index_list, fold_data_list, fold_label_list, test_fold_id=0, scaler_save_path=None):
        """Prepare train/val/test splits."""
        self.num_folds = 5
        assert 0 <= test_fold_id < self.num_folds

        train_data_list = [fold_data_list[i] for i in range(self.num_folds) if i != test_fold_id]
        train_label_list = [fold_label_list[i] for i in range(self.num_folds) if i != test_fold_id]
        train_index_list = [fold_index_list[i] for i in range(self.num_folds) if i != test_fold_id]

        train_data = np.concatenate(train_data_list, axis=0)
        train_label = np.concatenate(train_label_list, axis=0)
        train_index = np.concatenate(train_index_list, axis=0)

        idx = np.arange(len(train_data))
        np.random.shuffle(idx)
        split = int(0.9 * len(idx))
        tr_idx, val_idx = idx[:split], idx[split:]

        train_data_split = train_data[tr_idx]
        train_label_split = train_label[tr_idx]
        train_index_split = train_index[tr_idx]

        val_data_split = train_data[val_idx]
        val_label_split = train_label[val_idx]
        val_index_split = train_index[val_idx]

        scaler = StandardScaler()
        ori_shape_train = train_data_split.shape
        train_data_split_flat = train_data_split.reshape(-1, ori_shape_train[-1])
        scaler.fit(train_data_split_flat)
        if scaler_save_path:
            with open(os.path.join(scaler_save_path, 'scaler.pkl'), 'wb') as f:
                pickle.dump(scaler, f)
            print_aptxt(f"Scaler saved to {scaler_save_path}/scaler.pkl", Exp_log_path)

        train_data_split = scaler.transform(train_data_split_flat).reshape(ori_shape_train)
        ori_shape_val = val_data_split.shape
        val_data_split = scaler.transform(val_data_split.reshape(-1, ori_shape_val[-1])).reshape(ori_shape_val)

        def make_loader(data, label, index):
            comb = list(zip(data, label, index))
            np.random.shuffle(comb)
            data, label, index = zip(*comb)
            data = np.array(data)
            label = np.array(label)
            index = np.array(index)

            batch_size = self.args.batch_size
            batches = []
            for i in range(0, len(data), batch_size):
                end = min(i + batch_size, len(data))
                pkg = dotdict()
                pkg.batch_x = torch.tensor(data[i:end], dtype=torch.float32)
                pkg.batch_y = torch.tensor(label[i:end], dtype=torch.float32)
                pkg.batch_ids = index[i:end]
                batches.append(pkg)
            return batches

        self.train_loader = make_loader(train_data_split, train_label_split, train_index_split)
        self.vali_loader = make_loader(val_data_split, val_label_split, val_index_split)

        test_data = fold_data_list[test_fold_id]
        test_label = fold_label_list[test_fold_id]
        test_index = fold_index_list[test_fold_id]
        test_shape = test_data.shape
        test_data = scaler.transform(test_data.reshape(-1, test_shape[-1])).reshape(test_shape)
        self.test_loader = make_loader(test_data, test_label, test_index)

        self.scaler = scaler
        self.have_got_dataset = True

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        print_aptxt("Optimizer: Adam", Exp_log_path)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        print_aptxt("Loss function: CrossEntropyLoss", Exp_log_path)
        return criterion

    def _select_distillation_criterion(self):
        criterion = nn.KLDivLoss(reduction='batchmean')
        print_aptxt("Distillation loss: KLDivLoss", Exp_log_path)
        return criterion

    def _predict(self, batch_x):
        """Student forward pass. Returns (output, feature, keyness)."""

        def _run_model():
            outputs = self.model(batch_x)
            # KEL models return: (pred_logits, features, keyness_vector)
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()

        return outputs  # (pred, feature, keyness)

    def calculate_metrics(self, predictions, labels):
        """Calculate AD metrics including FNR."""
        pred_indices = torch.argmax(predictions, dim=1)
        label_indices = torch.argmax(labels, dim=1)

        accuracy = accuracy_score(label_indices.cpu().numpy(), pred_indices.cpu().numpy())
        cm = confusion_matrix(label_indices.cpu().numpy(), pred_indices.cpu().numpy())
        precision = precision_score(label_indices.cpu().numpy(), pred_indices.cpu().numpy(), average='macro')
        recall = recall_score(label_indices.cpu().numpy(), pred_indices.cpu().numpy(), average='macro')
        f1 = f1_score(label_indices.cpu().numpy(), pred_indices.cpu().numpy(), average='macro')

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            FNR = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        else:
            FNR = 1.0 - recall

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'FNR': FNR,
            'confusion_matrix': cm
        }

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for vali_sam in vali_loader:
                batch_x = vali_sam.batch_x.float().to(self.device)
                batch_y = vali_sam.batch_y.float().to(self.device)

                outputs, _, _ = self._predict(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        avg_loss = np.average(total_loss)
        self.model.train()
        return avg_loss

    def train(self, setting):
        """Main KEL training loop with dynamic alpha distillation."""
        self._init_model()
        path = os.path.join(f"{PROJECT_ROOT}/ModelCheckpoints", self.args.checkpoints, "KEL_AD", setting)
        if self.args.save_model_path:
            path = os.path.join(f"{PROJECT_ROOT}/ModelCheckpoints", self.args.save_model_path)
        os.makedirs(path, exist_ok=True)
        self._get_data(fold_index_list, fold_data_list, fold_label_list, self.args.testfoldid, scaler_save_path=path)

        train_loader, vali_loader, test_loader = self.train_loader, self.vali_loader, self.test_loader

        print_aptxt(f"Train samples: {len(train_loader)}", Exp_log_path)
        print_aptxt(f"Val samples: {len(vali_loader)}", Exp_log_path)
        print_aptxt(f"Test samples: {len(test_loader)}", Exp_log_path)


        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        last_lr = self.args.learning_rate

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        self._check_model_arcsize()
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        distill_criterion = self._select_distillation_criterion()

        # Training efficiency tracking
        total_train_start_time = time.time()
        epoch_times = []

        for epoch in range(self.args.train_epochs):
            print_aptxt(f"Starting epoch {epoch + 1}:", Exp_log_path)
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_start_time = time.time()

            for i, train_sam in enumerate(train_loader):
                iter_count += 1
                batch_x = train_sam.batch_x.float().to(self.device)
                batch_y = train_sam.batch_y.float().to(self.device)
                model_optim.zero_grad()

                # Teacher forward (frozen)
                with torch.no_grad():
                    teacher_outputs, teacher_feature = self.teacher_model(batch_x)

                # Student forward
                student_outputs, student_feature, student_keyness = self._predict(batch_x)

                # Dynamic alpha based on loss ratio
                teacher_ce_loss = criterion(teacher_outputs, batch_y)
                student_ce_loss = criterion(student_outputs, batch_y)
                alpha = torch.abs(teacher_ce_loss / (student_ce_loss + 1e-8) - 1)

                # Distillation loss (KL divergence on features)
                distill_loss = distill_criterion(
                    F.log_softmax(student_feature / self.args.temperature, dim=1),
                    F.softmax(teacher_feature / self.args.temperature, dim=1)
                )

                # Total loss: alpha * distillation + task loss
                loss = alpha * distill_loss + student_ce_loss
                train_loss.append(loss.item())

                if (i + 1) % 200 == 0:
                    print_aptxt(
                        f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}",
                        Exp_log_path
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print_aptxt(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s', Exp_log_path)
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print_aptxt(
                f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}",
                Exp_log_path
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print_aptxt("Early stopping triggered", Exp_log_path)
                break

            if self.args.lradj == 'type3':
                last_lr = adjust_learning_rate(model_optim, epoch + 1, self.args,
                                               early_stopping.is_stop_descending, last_lr)
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        # Save efficiency metrics
        total_train_time = time.time() - total_train_start_time
        avg_epoch_time = np.mean(epoch_times) if epoch_times else 0.0

        self.train_efficiency_metrics = {
            'ET': avg_epoch_time,
            'TTT': total_train_time
        }

        print_aptxt(f"Training Efficiency - ET: {avg_epoch_time:.4f}s, TTT: {total_train_time:.4f}s", Exp_log_path)

        self.test_clas()
        torch.cuda.empty_cache()

    def test_clas(self):
        """Testing with keyness extraction and inference time measurement."""
        if not self.have_got_dataset:
            self._get_data(fold_index_list, fold_data_list, fold_label_list, self.args.testfoldid)

        indexes = []
        pred_out = []
        true_out = []
        keyness_list = []

        target_samples = 32
        self.model.eval()

        # GPU warmup
        with torch.no_grad():
            for test_sam in self.test_loader:
                if test_sam.batch_x.size(0) > 0:
                    _ = self._predict(test_sam.batch_x[:1].float().to(self.device))
                    break

        # Measure IT32
        sample_count = 0
        inference_start_time = time.time()
        with torch.no_grad():
            for test_sam in self.test_loader:
                if sample_count >= target_samples:
                    break
                batch_size = test_sam.batch_x.size(0)
                remaining = target_samples - sample_count
                batch_x = test_sam.batch_x[:remaining].float().to(self.device)
                _ = self._predict(batch_x)
                sample_count += batch_x.size(0)
                if sample_count >= target_samples:
                    break

        it32 = time.time() - inference_start_time if sample_count > 0 else 0.0
        tp = sample_count / it32 if it32 > 0 else 0.0

        # Full evaluation with keyness extraction
        with torch.no_grad():
            for test_sam in self.test_loader:
                outputs = self._predict(test_sam.batch_x.float().to(self.device))
                output, feature, keyness = outputs

                output = output.reshape(-1, self.args.StdntArgs.clasnum)
                result = torch.zeros_like(output)
                _, max_indices = torch.max(output, dim=1)
                result[torch.arange(result.size(0)), max_indices] = 1

                indexes.append(test_sam.batch_ids)
                pred_out.append(result)
                true_out.append(test_sam.batch_y)
                keyness_list.append(keyness)

            pred_out = torch.concat(pred_out, dim=0).reshape(-1, self.args.StdntArgs.clasnum)
            true_out = torch.concat(true_out, dim=0).reshape(-1, self.args.StdntArgs.clasnum)
            indexes = np.concatenate(indexes, axis=0)

            # Concatenate keyness vectors for interpretability analysis
            all_keyness = torch.concat(keyness_list, dim=0) if keyness_list else None

        metrics = self.calculate_metrics(pred_out, true_out)

        if hasattr(self, 'train_efficiency_metrics'):
            metrics.update(self.train_efficiency_metrics)
        metrics['IT32'] = it32
        metrics['TP'] = tp

        self.test_recorder(metrics, Exp_log_path)
        self.test_recorder(metrics, Exp_outcome_path, False)

    def test_recorder(self, metrics, target_file_path, is_print_out=True):
        if is_print_out:
            print_aptxt(f"ACC: {metrics.get('accuracy', 0):.4f}", target_file_path)
            print_aptxt(f"F1: {metrics.get('f1_score', 0):.4f}", target_file_path)
            print_aptxt(f"ET: {metrics.get('ET', 0):.4f}s", target_file_path)
            print_aptxt(f"TTT: {metrics.get('TTT', 0):.4f}s", target_file_path)
            print_aptxt(f"IT32: {metrics.get('IT32', 0):.4f}s", target_file_path)
            print_aptxt(f"TP: {metrics.get('TP', 0):.2f} samples/s", target_file_path)
            print_aptxt(f"FNR: {metrics.get('FNR', 0):.4f}", target_file_path)
        else:
            append_text_to_file(f"ACC: {metrics.get('accuracy', 0):.4f}", target_file_path)
            append_text_to_file(f"F1: {metrics.get('f1_score', 0):.4f}", target_file_path)
            append_text_to_file(f"ET: {metrics.get('ET', 0):.4f}s", target_file_path)
            append_text_to_file(f"TTT: {metrics.get('TTT', 0):.4f}s", target_file_path)
            append_text_to_file(f"IT32: {metrics.get('IT32', 0):.4f}s", target_file_path)
            append_text_to_file(f"TP: {metrics.get('TP', 0):.2f} samples/s", target_file_path)
            append_text_to_file(f"FNR: {metrics.get('FNR', 0):.4f}", target_file_path)


if __name__ == "__main__":
    args, setting = get_args()
    args_meanings = get_args_meanings()
    data_select_pattern = args.data_select_pattern

    check_and_generate_data(output_folder=f"{PROJECT_ROOT}\ProcessedData\ProcessedData_Diagnosis_DeD_{data_select_pattern}")

    Exp_log_dir, Exp_log_path, Exp_outcome_path = init_exp_log(
        method_info=args.model_name,
        data_info=args.data,
        task_type=f"KEL_AD_{data_select_pattern}"
    )

    append_text_to_file(str(args_meanings), Exp_log_path)
    print_aptxt('Args in experiment:' + str(args), Exp_log_path)

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    global fold_data_list, fold_label_list, fold_index_list
    fold_data_list, fold_label_list, fold_index_list = LoadDataFolds(data_select_pattern)

    exp = Exp_KEL(args)
    exp.train(setting)
    torch.cuda.empty_cache()