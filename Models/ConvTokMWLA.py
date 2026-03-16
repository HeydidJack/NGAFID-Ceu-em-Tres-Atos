import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from math import sqrt


class ConvTokenizer(nn.Module):
    def __init__(self, configs):
        super(ConvTokenizer, self).__init__()
        self.L_patch = configs.L_patch
        self.in_dim = configs.in_dim
        self.token_dim = configs.token_dim
        # Define convolution layer
        self.conv = nn.Conv1d(in_channels=self.in_dim,
                              out_channels=self.token_dim - 2 * self.in_dim - 1,  # Subtract dimensions for statistics and positional encoding
                              kernel_size=self.L_patch,
                              stride=self.L_patch)

    def forward(self, x):
        # Input x shape is (B, full_len, d)
        batch_size, full_len, dim = x.shape

        # Adjust input shape for convolution layer
        x = x.permute(0, 2, 1)  # Adjust to (B, d, full_len)

        # Perform convolution operation
        conv_out = self.conv(x)  # Output shape is (B, token_dim - 2 * d - 1, N_token)
        conv_out = conv_out.permute(0, 2, 1)  # Adjust to (B, N_token, token_dim - 2 * d - 1)

        # Calculate mean and std for each patch segment
        # Reshape input x to (B, N_token, L_patch, d)
        x = x.permute(0, 2, 1)  # Adjust back to (B, full_len, d)
        N_token = full_len // self.L_patch
        x = x.view(batch_size, N_token, self.L_patch, dim)

        # Calculate mean and std
        mean = x.mean(dim=2)  # Shape is (B, N_token, d)
        std = x.std(dim=2, unbiased=False)  # Shape is (B, N_token, d)

        # Concatenate mean and std to convolution features
        stats = torch.cat([mean, std], dim=-1)  # Shape is (B, N_token, 2 * d)

        # Generate positional encoding
        position_ids = torch.arange(N_token, device=x.device).unsqueeze(0).expand(batch_size, N_token).unsqueeze(-1)
        position_enc = torch.sin(position_ids * 0.1)  # Use simple sine function to generate positional encoding, shape is (B, N_token, 1)

        # Concatenate positional encoding to features
        tokens = torch.cat([conv_out, stats, position_enc], dim=-1)  # Shape is (B, N_token, token_dim)

        return tokens

class Lambda(torch.nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=1,
                                  padding_mode='circular')
        # Replace BatchNorm1d with LayerNorm
        self.layer_norm = nn.LayerNorm(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)  # Adjust dimensions for LayerNorm
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.maxPool(x.permute(0, 2, 1))
        x = x.transpose(1, 2)
        return x


# Masking
# Mask fill for upper triangular matrix
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():  # Use torch.no_grad() context manager, indicating no gradient calculation when generating mask. This is because the mask is a fixed tensor and does not need to be updated in backpropagation.
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property  # Define a read-only property mask for returning the generated mask.
    def mask(self):
        return self._mask


# General Encoder layer, just fill in the attention block when using
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


# General Encoder structure
class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


# General attention layer
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

# Self-Attention
# Full attention mechanism
# =================== Sliding Window Attention ===================
class MultiWindowAttention(nn.Module):
    def __init__(self, window_size=3, mask_flag=False, scale=None,
                 attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.w_size = window_size if isinstance(window_size, list) else [window_size]
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        assert L == S, "MultiWindowAttention requires L == S"
        device = queries.device

        # If only one int is given, replicate to all heads
        if len(self.w_size) == 1:
            self.w_size = self.w_size * H
        assert len(self.w_size) == H, f"window_size list length {len(self.w_size)} != n_heads {H}"

        scale = self.scale or 1.0 / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # (B, H, L, L)

        # Construct independent mask for each head
        for h in range(H):
            w = self.w_size[h]
            # Band mask
            band = torch.ones(L, L, dtype=torch.bool, device=device)
            band = torch.triu(band, diagonal=-w)
            band = torch.tril(band, diagonal=w)
            scores[:, h].masked_fill_(~band, -np.inf)

        # Additionally stack upper triangular causal mask (optional)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ConvTokMWLA(nn.Module):
    def __init__(self, configs):
        super(ConvTokMWLA, self).__init__()
        self.L_patch = configs.L_patch
        self.num_classes = configs.clasnum
        # Replace with convolutional tokenizer
        self.tokenizer = ConvTokenizer(configs)
        self.device = self._acquire_device(configs.use_gpu,
                                           configs.gpu,
                                           configs.use_multi_gpu,
                                           configs.devices)
        modules = OrderedDict()
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        MultiWindowAttention(
                            window_size=configs.viewindow_size,  # Supports list / int
                            mask_flag=False,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention),
                        configs.token_dim, configs.n_heads),
                    configs.token_dim,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.token_dim
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.token_dim)
        )
        modules['avg_pool'] = Lambda(f=lambda x: torch.mean(x, dim=-2))
        modules['linear'] = torch.nn.Linear(in_features=configs.token_dim,
                                            out_features=self.num_classes)

        self.model = torch.nn.Sequential(modules)

    def _acquire_device(self, use_gpu=True, gpu=0, use_multi_gpu=False, devices=[0]):
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu) if not use_multi_gpu else devices
            device = torch.device('cuda:{}'.format(gpu))
        else:
            device = torch.device('cpu')
        return device

    def forward(self, x):
        batchnum, full_len, dim = x.shape
        # Ensure input length is integer multiple of L_patch
        padding_size = (self.L_patch - (full_len % self.L_patch)) % self.L_patch
        if padding_size > 0:
            padding = torch.zeros((batchnum, padding_size, dim), dtype=x.dtype, device=x.device)
            x = torch.cat((x, padding), dim=1)
        # Use convolutional tokenizer
        tokens = self.tokenizer(x)
        enc_out, _ = self.encoder(tokens)
        enc_out = self.model.get_submodule('avg_pool')(enc_out)
        y = self.model.get_submodule('linear')(enc_out)
        return y, enc_out

"""
def get_args():
    args = dotdict()
    args.gpu = 0
    args.lradj = 'type3'
    args.devices = '0'
    args.use_gpu = True
    args.use_multi_gpu = False
    if args.use_gpu and args.use_multi_gpu:  # Multi-GPU usage check
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    args.model_name = "ConvTokMWLA"
    args.data = 'NGAFID'
    args.model = ModelFactory().get_model_class(args.model_name)
    args.checkpoints = f'./{args.data}/{args.model_name}/my_checkpoints/'
    args.L_patch = 4
    args.clasnum = 2
    args.full_len = 2048
    args.e_layers = 2
    args.dropout = 0.01
    args.n_heads = 4
    args.viewindow_size = [1, 3, 5, 0]
    args.in_dim = 23
    args.d_ff = 512
    args.factor = 5
    args.token_dim = 128
    args.output_attention = False
    args.patience = 3
    args.learning_rate = 1e-4
    args.batch_size = 32
    args.use_amp = False
    args.activation = 'gelu'
    args.train_epochs = 100
    args.testfoldid = 1
    # Set model save path
    setting = '{}_{}_lp{}_td{}_nh{}_el{}_vw{}_df{}'.format(
        args.model_name,
        args.data,
        args.L_patch,
        args.token_dim,
        args.n_heads,
        args.e_layers,
        args.viewindow,
        args.d_ff, 0)
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
        "L_patch": "Sequence length for each token when tokenizing",
        "clasnum": "Number of classes for classification task.",
        "e_layers": "Number of encoder layers.",
        "dropout": "Dropout ratio.",
        "n_heads": "Number of heads in multi-head attention mechanism.",
        "viewindow_size": "Attention window size list.",
        "in_dim": "Dimension of input features.",
        "hidden_dim": "Dimension of hidden layers.",
        "d_ff": "Dimension of feed-forward network.",
        "output_attention": "Whether to output attention weights.",
        "patience": "Patience value for early stopping mechanism.",
        "learning_rate": "Learning rate.",
        "batch_size": "Size of each batch.",
        "token_dim": "Token dimension.",
        "use_amp": "Whether to use Automatic Mixed Precision (AMP).",
        "activation":'Activation function in Encoder.',
        "train_epochs": "Total number of training epochs."
    }
    return args_meanings_dict
"""