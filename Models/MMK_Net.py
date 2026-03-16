import torch
from collections import OrderedDict

import torch
from collections import OrderedDict

class MMK_Block(torch.nn.Module):
    def __init__(self, input_size, filters):
        super(MMK_Block, self).__init__()

        self.bottleneck1 = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )

        self.conv1 = torch.nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )

        self.conv3 = torch.nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=3,
            stride=1,
            padding='same',
            bias=False
        )

        self.conv5 = torch.nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=5,
            stride=1,
            padding='same',
            bias=False
        )

        # Replace BatchNorm1d with LayerNorm
        self.layer_norm = torch.nn.LayerNorm(
            normalized_shape=3 * filters,
            elementwise_affine=True
        )

    def forward(self, x):
        x0 = self.bottleneck1(x)
        x1 = self.conv1(x0)
        x2 = self.conv3(x0)
        x3 = self.conv5(x0)
        y = torch.concat([x1, x2, x3], dim=1)
        y = torch.nn.functional.relu(self.layer_norm(y.permute(0, 2, 1)).permute(0, 2, 1))
        return y


class Residual(torch.nn.Module):
    def __init__(self, input_size, filters):
        super(Residual, self).__init__()

        self.bottleneck = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=3 * filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )

        # Replace BatchNorm1d with LayerNorm
        self.layer_norm = torch.nn.LayerNorm(
            normalized_shape=3 * filters,
            elementwise_affine=True
        )

    def forward(self, x, y):
        y = y + self.layer_norm(self.bottleneck(x).permute(0, 2, 1)).permute(0, 2, 1)
        y = torch.nn.functional.relu(y)
        return y


class Lambda(torch.nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class MMK_Net(torch.nn.Module):
    def __init__(self, configs):
        super(MMK_Net, self).__init__()

        self.input_size = configs.in_dim
        self.num_classes = configs.clasnum
        self.filters = configs.filters
        self.depth = configs.inception_layers

        modules = OrderedDict()

        for d in range(self.depth):
            modules[f'mmk_block_{d}'] = MMK_Block(
                input_size=self.input_size if d == 0 else 3 * self.filters,
                filters=self.filters,
            )
            if d % 3 == 2:
                modules[f'residual_{d}'] = Residual(
                    input_size=self.input_size if d == 2 else 3 * self.filters,
                    filters=self.filters,
                )

        modules['avg_pool'] = Lambda(f=lambda x: torch.mean(x, dim=-1))
        modules['linear'] = torch.nn.Linear(in_features=3 * self.filters, out_features=self.num_classes)

        self.model = torch.nn.Sequential(modules)

    def forward(self, x):
        x = x.transpose(1, 2)
        for d in range(self.depth):
            y = self.model.get_submodule(f'mmk_block_{d}')(x if d == 0 else y)
            if d % 3 == 2:
                y = self.model.get_submodule(f'residual_{d}')(x, y)
                x = y
        y = self.model.get_submodule('avg_pool')(y)
        out = self.model.get_submodule('linear')(y)
        return out, y



"""
def get_args():
    args = dotdict()
    args.gpu = 0
    args.lradj = 'type3'
    args.devices = '0'
    args.use_gpu = True
    args.use_multi_gpu = False
    args.model_name = "MMK_Net"
    args.model = ModelFactory().get_model_class(args.model_name)
    args.data = "NGAFID"
    args.checkpoints = f'./{args.data}/{args.model_name}/my_checkpoints/'
    args.inception_layers = 4
    args.clasnum = 19
    args.full_len = 2048
    args.dropout = 0.01
    args.in_dim = 23
    args.hidden_dim = 2048
    args.output_attention = False
    args.patience = 3
    args.learning_rate = 1e-4
    args.batch_size = 32
    args.filters = 256
    args.use_amp = False
    args.activation = 'gelu'
    args.train_epochs = 1000
    args.testfoldid = 1
    # Set model save path
    setting = '{}_{}_il{}_ft{}'.format(
        args.model_name,
        args.data,
        args.inception_layers,
        args.filters, 0)
    return args, setting

def get_args_meanings_dict():
    args_meanings_dict = {
        "gpu": "GPU to use.",
        "lradj": "Learning rate adjustment strategy, type3 means halving the learning rate whenever loss stops decreasing.",
        "devices": "Description of the GPU being used.",
        "use_gpu": "Description of whether to use GPU.",
        "use_multi_gpu": "Whether to use multiple GPUs.",
        "checkpoints": "Save path for model checkpoints.",
        "model": "Model class being used.",
        "model_name": "Name of the model.",
        "data": "Dataset being used.",
        "inception_layers": "Number of Inception module layers.",
        "clasnum": "Number of classes for classification task.",
        "dropout": "Dropout ratio.",
        "n_heads": "Number of heads in multi-head attention mechanism.",
        "in_dim": "Dimension of input features.",
        "patience": "Patience value for early stopping mechanism.",
        "learning_rate": "Learning rate.",
        "batch_size": "Size of each batch.",
        "filters": "Number of convolutional filters.",
        "use_amp": "Whether to use Automatic Mixed Precision (AMP).",
        "activation": "Activation function in Encoder.",
        "train_epochs": "Total number of training epochs."
    }
    return args_meanings_dict
"""