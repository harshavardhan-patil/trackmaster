import argparse
import time
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
import tmrl


# ============================================================
# CNN Architecture Configuration
# ============================================================
# Default TMRL configuration: 4 grayscale images of 64 x 64 pixels
imgs_buf_len = 4  # Number of stacked grayscale images
img_height = 64   # Image height in pixels
img_width = 64    # Image width in pixels


# ============================================================
# Helper Functions for CNN
# ============================================================

def conv2d_out_dims(conv_layer, h_in, w_in):
    """
    Calculate output dimensions of a Conv2d layer

    Args:
        conv_layer: nn.Conv2d layer
        h_in: input height
        w_in: input width

    Returns:
        (h_out, w_out): output dimensions
    """
    h_out = (h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) // conv_layer.stride[0] + 1
    w_out = (w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) // conv_layer.stride[1] + 1
    return h_out, w_out


def num_flat_features(x):
    """
    Calculate number of features in a flattened tensor

    Args:
        x: input tensor

    Returns:
        number of features
    """
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Create a multi-layer perceptron

    Args:
        sizes: list of layer sizes
        activation: activation function
        output_activation: activation for output layer

    Returns:
        nn.Sequential MLP
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class TrackMasterCNN(nn.Module):
    """
    CNN (Convolutional Neural Network) model for processing TMRL observations
    """

    def __init__(self):
        super(TrackMasterCNN, self).__init__()
        # Convolutional layers processing screenshots:
        # The default config.json gives 4 grayscale images of 64 x 64 pixels
        self.h_out, self.w_out = img_height, img_width
        self.conv1 = nn.Conv2d(imgs_buf_len, 64, 8, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)
        self.conv4 = nn.Conv2d(128, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)
        self.out_channels = self.conv4.out_channels

        # Dimensionality of the CNN output:
        self.flat_features = self.out_channels * self.h_out * self.w_out

        # Dimensionality of the MLP input:
        # The MLP input will be formed of:
        # - the flattened CNN output
        # - the current speed, gear and RPM measurements (3 floats)
        # - the 2 previous actions (2 x 3 floats), important because of the real-time nature of our controller
        # - when the module is the critic, the selected action (3 floats)
        float_features = 9
        self.mlp_input_features = self.flat_features + float_features

        # MLP layers:
        # (when using the model as a policy, we will sample from a multivariate gaussian defined later;
        # thus, the output dimensionality is 1 for the critic, and we will define the output layer of policies later)
        self.mlp_layers = [256, 256]
        self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU)

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x: input tuple (speed, gear, rpm, images, act1, act2) or
               (speed, gear, rpm, images, act1, act2, act) for critic

        Returns:
            output tensor
        """
        speed, gear, rpm, images, act1, act2 = x

        speed = speed / 1000.0  # Max speed approx 1000
        rpm = rpm / 10000.0     # Max RPM approx 10000
        gear = gear / 6.0       # Max gear 6
        images = images / 255.0

        # we will stack these greyscale images along the channel dimension of our input tensor)
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Now we will flatten our output feature map.
        # Let us double-check that our dimensions are what we expect them to be:
        flat_features = num_flat_features(x)
        assert flat_features == self.flat_features, f"x.shape:{x.shape}, flat_features:{flat_features}, self.out_channels:{self.out_channels}, self.h_out:{self.h_out}, self.w_out:{self.w_out}"

        # All good, let us flatten our output feature map:
        x = x.view(-1, flat_features)

        # Concat and feed the result along our float values to the MLP:
        x = torch.cat((speed, gear, rpm, x, act1, act2), -1)
        x = self.mlp(x)

        # And this gives us the output of our deep neural network :)
        return x
