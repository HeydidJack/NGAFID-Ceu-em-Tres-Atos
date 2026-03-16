import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()
        self.in_len = configs.in_len
        self.input_dim = configs.in_dim
        self.hidden_dim = configs.d_model
        self.num_classes = configs.clasnum
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.conv1 = nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=self.kernel_size,
                               stride=self.stride, padding=self.padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(self.hidden_dim * (self.in_len // 2), self.num_classes)  # Assume length is halved after pooling

    def forward(self, x):
        # Input x shape is (B, L, D), need to adjust to (B, D, L) for Conv1d
        x = x.transpose(1, 2)  # Adjust dimensions to (B, D, L)
        x = self.conv1(x)  # Conv operation, output shape is (B, hidden_dim, L')
        x = self.relu(x)  # Activation function
        x = self.pool(x)  # Pooling operation, output shape is (B, hidden_dim, L'')
        # Flatten
        x = x.view(x.size(0), -1)  # Flatten to (B, hidden_dim * L'')
        # Fully connected layer
        out = self.fc(x)  # Output shape is (B, N_Clas)
        return out, x




"""
def get_args():
    args = dotdict()
    args.gpu = 0
    args.lradj = 'type3'
    args.devices = '0'
    args.use_gpu = True
    args.use_multi_gpu = False
    args.model = ModelFactory().get_model_class("CNN")
    args.model_name = "CNN"
    args.data = 'NGAFID'
    args.checkpoints = f'./{args.data}/{args.model_name}/my_checkpoints/'
    args.in_len = 2048
    args.in_dim = 23
    args.clasnum = 2
    args.patience = 3
    args.d_model = 128
    args.learning_rate = 1e-5
    args.batch_size = 256
    args.train_epochs = 200
    args.testfoldid = 1
    # Set model save path
    setting = '{}_{}_dm{}_cn{}'.format(
        args.model_name,
        args.data,
        args.d_model,
        args.clasnum, 0)
    return args, setting

def get_args_meanings_dict():
    # Create dictionary
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
        "in_len":"Input sequence length.",
        "in_dim": "Dimension of input features.",
        "d_model": "Dimension of hidden layers.",
        "patience": "Patience value for early stopping mechanism.",
        "learning_rate": "Learning rate.",
        "batch_size": "Size of each batch.",
        "train_epochs": "Total number of training epochs."
    }
"""