import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assume ConvTokMHSA and MMK_Net models are already defined
from Models.ConvTokMHSA import ConvTokMHSA as ConvTokMHSA
from Models.MMK_Net import MMK_Net


class LMSD(nn.Module):
    def __init__(self, configs, m=0):
        super(LMSD, self).__init__()
        self.clasnum = configs.clasnum
        self.m = configs.NormalIndex if configs.NormalIndex else m  # Specify dimension index of normal class
        configs.clasnum = configs.deteclasnum
        self.conv_tok_mhsa = ConvTokMHSA(configs)  # Load ConvTokMHSA model
        self.conv_tok_mhsa.load_state_dict(torch.load(configs.detecprepath))
        for param in self.conv_tok_mhsa.parameters():
            param.requires_grad = False
        configs.clasnum = configs.frclasnum
        self.itfilters = configs.filters
        self.mmk_net = MMK_Net(configs)  # Load MMK_Net model
        self.mmk_net.load_state_dict(torch.load(configs.frprepath))
        for param in self.mmk_net.parameters():
            param.requires_grad = False
        configs.clasnum = self.clasnum

    def forward(self, x):
        # Input x shape is (B, L, D)
        # Get ConvTokMHSA output, shape is (B, 2)
        conv_out, cmf = self.conv_tok_mhsa(x)

        # Initialize final output, shape is (B, clasnum)
        final_output = torch.zeros((conv_out.size(0), self.clasnum), device=conv_out.device)

        # Determine if healthy
        is_healthy = conv_out[:, 0] > conv_out[:, 1]

        # 3. Build healthy sample output (all branches)
        # Dimension m=healthy score, other dimensions=-inf
        final_output[is_healthy, self.m] = conv_out[is_healthy, 0]
        final_output[is_healthy, :self.m] = -float('inf')
        final_output[is_healthy, self.m + 1:] = -float('inf')

        # If not healthy, set dimension m to negative infinity, other clasnum-1 dimensions filled by MMK_Net inference
        not_healthy = ~is_healthy
        if not_healthy.any():
            inception_out, itf = self.mmk_net(x[not_healthy])
            final_output[not_healthy, self.m] = -float('inf')  # Set dimension m to negative infinity
            final_output[not_healthy, :self.m] = inception_out[:, :self.m]
            final_output[not_healthy, self.m + 1:] = inception_out[:, self.m:]

            # Create a zero tensor, shape is (B, 3 * self.itfilters)
            itf_full = torch.zeros((conv_out.size(0), 3 * self.itfilters), device=conv_out.device)
            # Fill inference results to unhealthy sample positions
            itf_full[not_healthy] = itf
        else:
            # If all samples are healthy, create a zero itf tensor
            itf_full = torch.zeros((conv_out.size(0), 3 * self.itfilters), device=conv_out.device)

        # Concatenate features
        return torch.softmax(final_output, -1), torch.concat([cmf, itf_full], dim=-1)


"""
def get_args():
    args = dotdict()
    args.gpu = 0
    args.lradj = 'type3'
    args.devices = '0'
    args.use_gpu = True
    args.use_multi_gpu = False
    args.NormalIndex = 1
    args.model_name = "LMSD"
    args.data = 'NGAFID'
    args.model = ModelFactory().get_model_class(args.model_name)
    args.checkpoints = f'/{args.data}/{args.model_name}/my_checkpoints/'
    args.clasnum = 20
    args.deteclasnum = 2
    args.frclasnum = 19
    args.L_patch = 4
    args.full_len = 2048
    args.e_layers = 2
    args.dropout = 0.01
    args.n_heads = 2
    args.in_dim = 23
    args.d_ff = 256
    args.factor = 5
    args.token_dim = 128
    args.inception_layers = 4
    args.dropout = 0.01
    args.hidden_dim = 2048
    args.output_attention = False
    args.patience = 3
    args.learning_rate = 3e-6
    args.batch_size = 32
    args.filters = 256
    args.use_amp = False
    args.activation = 'gelu'
    args.train_epochs = 1
    args.testfoldid = 4
    args.detecprepath = "ModelCheckpoints/NGAFID/ConvTokMHSA/my_checkpoints/Detection/detec4dg_ConvTokMHSA_NGAFID_lp4_td128_nh4_el2_df512/checkpoint.pth"
    args.frprepath = "ModelCheckpoints/NGAFID/MMK_NetwoPool/my_checkpoints/FaultRecognization/fr4dg_MMK_NetwoPool_NGAFID_il4_ft256/checkpoint.pth"
    # Set model save path
    setting = '{}_{}_lp{}_td{}_el{}_il{}_ft{}_fold{}'.format(
        args.model_name,
        args.data,
        args.L_patch,  # L_patch
        args.token_dim,  # token_dim
        args.e_layers,  # encoder layers
        args.inception_layers,  # inception layers
        args.filters,  # filters
        args.testfoldid  # test fold ID
    )
    return args, setting

def get_args_meanings_dict():
    args_meanings_dict = {
        "gpu": "Specify GPU device ID to use.",
        "lradj": "Learning rate adjustment strategy, 'type3' means halving the learning rate when Loss stops decreasing.",
        "devices": "GPU device IDs used, supports description of single or multiple GPUs.",
        "use_gpu": "Whether to use GPU for training.",
        "use_multi_gpu": "Whether to use multi-GPU for distributed training.",
        "NormalIndex": "Normal class index number, used to specify normal class in specific dimension of model output.",
        "model_name": "Model name, used for identification and saving model.",
        "data": "Dataset name used.",
        "model": "Model class instantiation object, obtained through ModelFactory.",
        "checkpoints": "Save path for model checkpoints.",
        "L_patch": "Sequence length for each token when tokenizing.",
        "clasnum": "Number of classes for classification task.",
        "deteclasnum": "Number of classes for anomaly detection task.",
        "frclasnum": "Number of classes for fault recognition task.",
        "full_len": "Full length of input sequence.",
        "e_layers": "Encoder layers, represents number of layers in Encoder.",
        "dropout": "Dropout ratio, used for regularization to prevent overfitting.",
        "n_heads": "Number of heads in multi-head attention, used for parallel attention calculation.",
        "in_dim": "Input feature dimension, i.e., number of features per timestep.",
        "d_ff": "Feed-Forward Network dimension, used for intermediate layers in Encoder.",
        "factor": "Scaling factor in attention mechanism, used to adjust attention weight calculation.",
        "token_dim": "Token dimension, represents embedding dimension of each token.",
        "inception_layers": "Inception module layers, used for InceptionTime architecture.",
        "hidden_dim": "Hidden layer dimension, used for intermediate representation inside model.",
        "output_attention": "Whether to output attention weights, used for debugging or visualization.",
        "patience": "Early stopping patience, indicates how many epochs without improvement before stopping training.",
        "learning_rate": "Initial learning rate used during training.",
        "batch_size": "Batch size, used for training and validation.",
        "filters": "Number of convolutional filters, used for InceptionTime architecture.",
        "use_amp": "Whether to use Automatic Mixed Precision (AMP) training to accelerate training and reduce memory usage.",
        "activation": "Activation function, used for nonlinear transformation in Encoder.",
        "train_epochs": "Total training rounds, i.e., number of epochs for model training.",
        "testfoldid": "Test data fold number, used for cross-validation."
    }
    return args_meanings_dict
"""