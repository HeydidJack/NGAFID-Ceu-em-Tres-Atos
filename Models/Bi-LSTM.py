# Bi-LSTM.py
import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, configs):
        super(BiLSTM, self).__init__()
        self.input_dim   = configs.in_dim
        self.hidden_dim  = configs.d_model
        self.num_layers  = configs.num_layers
        self.num_classes = configs.clasnum
        self.bidirectional = getattr(configs, 'bidirectional', True)

        # 1x1 conv to compress channels -> reduce subsequent LSTM parameters
        self.proj = nn.Conv1d(self.input_dim, 64, 1)

        # Deep Bi-LSTM + LayerNorm + Dropout
        self.lstm = nn.LSTM(64, self.hidden_dim, self.num_layers,
                            batch_first=True,
                            bidirectional=self.bidirectional,
                            dropout=0.3 if self.num_layers > 1 else 0.)

        # Last-output projection head
        dim = self.hidden_dim * (2 if self.bidirectional else 1)
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.3),
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        # x: (B, L, input_dim)
        x = x.transpose(1, 2)          # (B, input_dim, L)
        x = self.proj(x).transpose(1, 2)  # (B, L, 64)

        h, _ = self.lstm(x)            # (B, L, dim)

        # Take the last timestep
        out = h[:, -1]                 # (B, dim)

        return self.fc(out), out

"""
def get_args():
    args = dotdict()
    args.gpu = 0
    args.lradj = 'type3'
    args.devices = '0'
    args.use_gpu = True
    args.use_multi_gpu = False
    args.model_name = "Bi-LSTM"
    args.model = ModelFactory().get_model_class(args.model_name)
    args.data = 'NGAFID'
    args.checkpoints = f'./{args.data}/{args.model_name}_BiLSTM3L/my_checkpoints/'

    # ---------- Core ----------
    args.num_layers   = 3          # 3-layer Bi-LSTM
    args.in_dim       = 23
    args.d_model      = 256        # hidden 256
    args.bidirectional = True
    args.clasnum      = 2          # 2/19 switchable
    # ----------------------------

    args.patience     = 5          # Early stopping patience relaxed, deep layers need more training
    args.learning_rate = 1e-3      # cosine annealing
    args.batch_size   = 64
    args.train_epochs = 200
    args.testfoldid   = 2

    setting = '{}_{}_nl{}_dm{}_cn{}'.format(
        args.model_name,
        args.data,
        args.num_layers,
        args.d_model,
        args.clasnum)
    return args, setting


def get_args_meanings_dict():
    return {
        "gpu": "GPU to use.",
        "lradj": "Learning rate adjustment strategy, type3 means halving the learning rate whenever loss stops decreasing.",
        "devices": "Description of the GPU being used.",
        "use_gpu": "Description of whether to use GPU.",
        "use_multi_gpu": "Whether to use multiple GPUs.",
        "checkpoints": "Save path for model checkpoints.",
        "model": "Model class being used.",
        "model_name": "Model name (registered to ModelFactory).",
        "data": "Dataset being used.",
        "num_layers": "Bi-LSTM layers (3 layers optimal).",
        "in_dim": "Input feature dimension (number of sensors).",
        "d_model": "LSTM hidden dimension.",
        "patience": "Early stopping patience value.",
        "learning_rate": "Initial learning rate (1e-3 + cosine).",
        "batch_size": "Size of each batch.",
        "train_epochs": "Total number of training epochs.",
        "bidirectional": "Whether bidirectional LSTM.",
        "clasnum": "Number of classes (2 or 19)."
    }
"""