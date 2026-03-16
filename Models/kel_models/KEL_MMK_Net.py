import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from math import sqrt
from Models.MMK_Net import MMK_Net
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


class KeynessModule(nn.Module):
    def __init__(self, configs):
        super(KeynessModule, self).__init__()
        self.full_len = configs.full_len
        self.D = configs.in_dim  # Input feature dimension
        self.sigmoid = nn.Sigmoid()

        # Define CNN to analyze input data and generate Keyness
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.D, out_channels=64, kernel_size=3, padding=1),  # Preserve sequence length
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=self.D, kernel_size=3, padding=1),
            # Output channels same as input feature dimension
            nn.ReLU()
        )

    def forward(self, x):
        B, L, D = x.size()
        assert L == self.full_len, "Input length L must match full_len"

        # Transpose feature dimension to channel dimension to fit convolution operations
        x_transposed = x.transpose(1, 2)  # Shape becomes (B, D, L)

        # Use CNN to generate Keyness
        Keyness = self.cnn(x_transposed)  # Shape is (B, D, L)
        Keyness = Keyness.transpose(1, 2)  # Transpose back to (B, L, D)

        # Apply sigmoid to Keyness
        KeynessW = self.sigmoid(Keyness)

        # Element-wise multiply Keyness with input x
        weighted_x = x * KeynessW

        return weighted_x, Keyness


# Add KeynessModule before MMK_Net model
class KEL_MMK_Net(nn.Module):
    def __init__(self, configs, is_tokenized=False):
        super(KEL_MMK_Net, self).__init__()
        self.keyness_module = KeynessModule(configs)
        self.mmk = MMK_Net(configs)
        self.KeyWpath = f"ModelCheckpoints/{configs.checkpoints}/keyness.pkl"

    def forward(self, x):
        # First pass through KeynessModule
        weighted_x, Keyness = self.keyness_module(x)
        # Then through MMK_Net model
        output, encoder_output = self.mmk(weighted_x)
        return output, encoder_output, Keyness


class TimeKeynessModule(nn.Module):
    def __init__(self, configs):
        super(TimeKeynessModule, self).__init__()
        self.full_len = configs.full_len
        self.scale = configs.scale
        self.D = configs.in_dim  # Input feature dimension
        self.sigmoid = nn.Sigmoid()

        # Define CNN to predict keyness time matrix
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.D, out_channels=64, kernel_size=3, padding=1),  # Preserve sequence length
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=1, kernel_size=3, padding=1),  # Output channels is 1
            nn.ReLU()
        )

        # Ensure scale divides full_len
        assert self.full_len % self.scale == 0, "scale must divide full_len"

    def forward(self, x):
        B, L, D = x.size()
        assert L == self.full_len, "Input length L must match full_len"

        # Transpose feature dimension to channel dimension to fit convolution operations
        x_transposed = x.transpose(1, 2)  # Shape becomes (B, D, L)

        # Use CNN to predict keyness time matrix
        KeynessW = self.cnn(x_transposed)  # Shape is (B, 1, L)

        # Adjust keyness time matrix shape to (B, L // scale, 1)
        KeynessW = KeynessW[:, :, ::self.scale]  # Sample to get (B, 1, L // scale)
        KeynessTime = KeynessW.transpose(1, 2)

        # Apply sigmoid activation to KeynessW
        KeynessW = self.sigmoid(KeynessW)

        # Expand KeynessW to KeynessScore with shape (B, L, D)
        KeynessScore = KeynessW.repeat_interleave(self.scale, dim=2)  # Repeat scale times, shape becomes (B, 1, L)
        KeynessScore = KeynessScore.repeat(1, D, 1)  # Repeat D times, shape becomes (B, D, L)
        KeynessScore = KeynessScore.transpose(1, 2)  # Transpose to (B, L, D)

        # Ensure KeynessScore shape matches input x time and feature dimensions
        assert KeynessScore.shape == (B, L, D), "KeynessScore shape mismatch"

        # Element-wise multiply KeynessScore with input x
        weighted_x = x * KeynessScore

        return weighted_x, KeynessTime


# Add TimeKeynessModule before MMK_Net model
class TimeKEL_MMK_Net(nn.Module):
    def __init__(self, configs, is_tokenized=False):
        super(TimeKEL_MMK_Net, self).__init__()
        self.keyness_module = TimeKeynessModule(configs)
        self.mmk = MMK_Net(configs)
        self.KeyWpath = f"ModelCheckpoints/{configs.checkpoints}/keyness.pkl"

    def forward(self, x):
        # First pass through KeynessModule
        weighted_x, Keyness = self.keyness_module(x)
        # Then through MMK_Net model
        output, encoder_output = self.mmk(weighted_x)
        return output, encoder_output, Keyness