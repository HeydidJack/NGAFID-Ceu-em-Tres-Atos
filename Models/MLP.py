import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, configs):
        super(MLP, self).__init__()
        self.input_dim = configs.in_dim
        self.hidden_dim = configs.d_model
        self.num_classes = configs.clasnum
        self.flatten = nn.Flatten()  # Flatten input to (B, L * D)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)  # Input layer to hidden layer
        self.relu = nn.ReLU()  # Non-linear activation function
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)  # Hidden layer to output layer

    def forward(self, x):
        x = self.flatten(x)  # Flatten input
        x = self.fc1(x)  # Input layer to hidden layer
        x = self.relu(x)  # Non-linear activation function
        out = self.fc2(x)  # Hidden layer to output layer
        return out, x



"""
def get_args():
    args = dotdict()
    args.gpu = 0
    args.lradj = 'type3'
    args.devices = '0'
    args.use_gpu = True
    args.use_multi_gpu = False
    args.model = ModelFactory().get_model_class("MLP")
    args.model_name = "MLP"
    args.data = 'NGAFID'
    args.checkpoints = f'./{args.data}/{args.model_name}/my_checkpoints/'
    args.in_dim = 2048 * 23  # Input sequence length
    args.clasnum = 2  # Number of classes for prediction
    args.patience = 3
    args.d_model = 128
    args.learning_rate = 1e-5
    args.batch_size = 256
    args.train_epochs = 200
    args.testfoldid = 1
    # Set model save path
    setting = '{}_{}_dm{}'.format(
        args.model_name,
        args.data,
        args.d_model, 0)
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
        "in_dim": "Dimension of input features.",
        "d_model": "Dimension of hidden layers.",
        "patience": "Patience value for early stopping mechanism.",
        "learning_rate": "Learning rate.",
        "batch_size": "Size of each batch.",
        "train_epochs": "Total number of training epochs."
    }
"""