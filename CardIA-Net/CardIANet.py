"""
CardIA-Net: An Explainable Deep Learning Model for MI Detection with ECG Lead Optimization.

This module provides a reusable PyTorch implementation of the CardIA-Net model:
- Multi-scale 1D Inception modules with residual shortcut
- Channel-preserving Inception block (stack of Inception modules + residual add)
- Global average pooling head
- Lightweight vector attention over the pooled features
- Linear classifier

Typical input shape: (batch_size, num_leads, signal_length)

Author: A.A.M. Bulbul (original study and research code)
Refactoring (model-only): for GitHub release
License: Choose and add your preferred license (e.g., MIT)

PyTorch >= 1.10
"""

from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


num_epochs = 1000      # Number of training epochs
batch_size = 32        # Batch Size
num_incep_block = 6    # Number of Inception Blocks
Num_CH = 4             # Number of input ECG leads
lr = 0.0001            # Learning rate
sig_len = 300          # Number of data-points in each ECG segment @500Hz

#######################################################################################
## Define the Attention Mechanism Module
#######################################################################################

# Define the attention mechanism module
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        
        # Linear layer to compute attention scores, keeping the input dimension
        self.attention_weights = nn.Linear(input_dim, input_dim, bias=False)
        
        # Bias term for the addition step (b)
        self.bias = nn.Parameter(torch.zeros(input_dim))  # Bias term

    def forward(self, x):
        # Dot product between input and weight matrix (e_t = x · W)
        scores = self.attention_weights(x)  # Shape: (batch_size, input_dim)

        # Apply hyperbolic tangent activation (h_t = tanh(e_t))
        tanh_scores = torch.tanh(scores)

        # Add bias to the result (y_t = h_t + b)
        biased_scores = tanh_scores + self.bias

        # Softmax to calculate attention weights (a_t = softmax(y_t))
        attention_weights = torch.softmax(biased_scores, dim=-1)  # Shape: (batch_size, input_dim)

        # Element-wise multiplication of the input with the attention weights
        weighted_sum = x * attention_weights  # Element-wise multiplication

        return weighted_sum


#######################################################################################
## Define the CardIA-Net model
#######################################################################################

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

class InceptionModulePlus(nn.Module):
    def __init__(self, in_channels, num_kernels=32, bottleneck_channels=32):
        super(InceptionModulePlus, self).__init__()
        self.bottleneck = ConvBlock(in_channels, bottleneck_channels, kernel_size=1)
        self.convs = nn.ModuleList([
            ConvBlock(bottleneck_channels, num_kernels, kernel_size=39, padding=19),
            ConvBlock(bottleneck_channels, num_kernels, kernel_size=19, padding=9),
            ConvBlock(bottleneck_channels, num_kernels, kernel_size=9, padding=4)
        ])
        self.mp_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, bottleneck_channels, kernel_size=1)
        )
        self.concat = nn.Identity()
        self.norm = nn.BatchNorm1d(num_kernels * 3 + bottleneck_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x_bottleneck = self.bottleneck(x)
        conv_outputs = [conv(x_bottleneck) for conv in self.convs]
        mp_output = self.mp_conv(x)
        concatenated = torch.cat(conv_outputs + [mp_output], dim=1)
        concatenated = self.norm(concatenated)
        return self.act(concatenated)

class InceptionBlockPlus(nn.Module): 
    def __init__(self, in_channels, blocks=num_incep_block, num_kernels=32, bottleneck_channels=32):
        super(InceptionBlockPlus, self).__init__()
        self.inception = nn.ModuleList([InceptionModulePlus(in_channels if i == 0 else num_kernels * 3 + bottleneck_channels) for i in range(blocks)])
        self.shortcut = nn.ModuleList([
            ConvBlock(in_channels, num_kernels * 3 + bottleneck_channels, kernel_size=1),
            nn.BatchNorm1d(num_kernels * 3 + bottleneck_channels)
        ])
        self.act = nn.ModuleList([nn.ReLU(), nn.ReLU()])  
        self.add = torch.add  

    def forward(self, x):
        residual = x
        for i, module in enumerate(self.shortcut):
            residual = module(residual)
        for module in self.inception:
            x = module(x)
        x = self.add(x, residual)  
        for activation in self.act:
            x = activation(x)
        return x

class CardIANet(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 3):
        super(CardIANet, self).__init__()
        self.backbone = nn.Sequential(InceptionBlockPlus(in_channels=Num_CH))
        self.head = nn.Sequential(
            GAP1d(),
        )

        # Attention mechanism
        self.attention = Attention(input_dim=128)
        
        self.final = nn.Sequential(
            LinBnDrop(in_features=128, out_features=3, bias=True)
        )


    def forward(self, x):
        x = self.backbone(x)         # (B, 128, L)
        x = self.head(x)             # (B, 128)
        # Attention mechanism
        x = self.attention(x)        # (B, 128)
        return self.final(x)         # (B, C)


class GAP1d(nn.Module):
    def __init__(self):
        super(GAP1d, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.gap(x)
        return self.flatten(x)

class LinBnDrop(nn.Module):
    def __init__(self, in_features, out_features, bias):
        super(LinBnDrop, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        x = self.linear(x)
        return x

#######################################################################################
## Prints the CardIA-Net model structure
#######################################################################################
# ------------------------------
# Prints structure
# ------------------------------
if __name__ == "__main__":
    model = CardIANet(in_channels=Num_CH, num_classes=3)
    x = torch.randn(batch_size, Num_CH, sig_len)
    y = model(x)
    print(model)
    print("Input:", x.shape, "Output:", y.shape)
