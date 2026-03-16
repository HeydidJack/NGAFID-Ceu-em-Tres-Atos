import torch
from collections import OrderedDict

class Inception(torch.nn.Module):
    def __init__(self, input_size, filters):
        super(Inception, self).__init__()

        self.bottleneck1 = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )

        self.conv10 = torch.nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=10,
            stride=1,
            padding='same',
            bias=False
        )

        self.conv20 = torch.nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=20,
            stride=1,
            padding='same',
            bias=False
        )

        self.conv40 = torch.nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=40,
            stride=1,
            padding='same',
            bias=False
        )

        self.max_pool = torch.nn.MaxPool1d(
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.bottleneck2 = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )

        self.batch_norm = torch.nn.BatchNorm1d(
            num_features=4 * filters
        )

    def forward(self, x):
        x0 = self.bottleneck1(x)
        x1 = self.conv10(x0)
        x2 = self.conv20(x0)
        x3 = self.conv40(x0)
        x4 = self.bottleneck2(self.max_pool(x))
        y = torch.concat([x1, x2, x3, x4], dim=1)
        y = torch.nn.functional.relu(self.batch_norm(y))
        return y


class Residual(torch.nn.Module):
    def __init__(self, input_size, filters):
        super(Residual, self).__init__()

        self.bottleneck = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=4 * filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )

        self.batch_norm = torch.nn.BatchNorm1d(
            num_features=4 * filters
        )

    def forward(self, x, y):
        y = y + self.batch_norm(self.bottleneck(x))
        y = torch.nn.functional.relu(y)
        return y


class Lambda(torch.nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class InceptionTime(torch.nn.Module):
    def __init__(self, configs):
        super(InceptionTime, self).__init__()

        self.input_size = configs.in_dim
        self.num_classes = configs.clasnum
        self.filters = configs.filters
        self.depth = configs.inception_layers

        modules = OrderedDict()

        for d in range(self.depth):
            modules[f'inception_{d}'] = Inception(
                input_size=self.input_size if d == 0 else 4 * self.filters,
                filters=self.filters,
            )
            if d % 3 == 2:
                modules[f'residual_{d}'] = Residual(
                    input_size=self.input_size if d == 2 else 4 * self.filters,
                    filters=self.filters,
                )

        modules['avg_pool'] = Lambda(f=lambda x: torch.mean(x, dim=-1))
        modules['linear'] = torch.nn.Linear(in_features=4 * self.filters, out_features=self.num_classes)

        self.model = torch.nn.Sequential(modules)

    def forward(self, x):
        x = x.transpose(1, 2)
        for d in range(self.depth):
            y = self.model.get_submodule(f'inception_{d}')(x if d == 0 else y)
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
    args.model_name = "InceptionTime"
    args.model = ModelFactory().get_model_class(args.model_name)
    args.data = "NGAFID"
    args.checkpoints = f'./{args.data}/{args.model_name}/my_checkpoints/'
    args.inception_layers = 4
    args.clasnum = 2
    args.dropout = 0.01
    args.in_dim = 23
    args.hidden_dim = 2048
    args.output_attention = False
    args.patience = 3
    args.learning_rate = 3e-5
    args.batch_size = 32
    args.filters = 128
    args.use_amp = False
    args.activation = 'gelu'
    args.train_epochs = 1
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