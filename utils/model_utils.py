"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1

A library of model classes that can used in Self-Supervised framework. The list of models:

CPC:          High-level model that contains Encoder and Autoregressive networks.
CPCEncoder:   CNN-based encoder
"""

import copy
import torch as th
from torch import nn


class CPC(nn.Module):
    """
    :param dict options: Generic dictionary to configure the model for training.
    :return: output of encoder, predictions computed from context and hidden (h) from GRU.

    Core model for contrastive predictive coding.
    """
    def __init__(self, options):
        super(CPC, self).__init__()
        # Get configuration
        self.options = options
        # Get the dimensions of all layers
        self.dims = options["conv_dims"]
        # Number of time steps, and batch size
        self.ts, self.bs = options["time_steps"], options["batch_size"]
        # Highest index to use when sampling time points from sequence
        self.hindex = int(self.options["sequence_length"] / self.options["downsampling_factor"]) - self.ts
        # Initialize online network
        self.encoder = CPCEncoder(options)
        # Initialize Sequence model: GRU expect (batch, seq, feature) when batch_first=True
        self.gru = nn.GRU(input_size=self.dims[-1], hidden_size=self.dims[-1], num_layers=2, batch_first=True)
        # Generate linear weights for each of time steps that we will make predictions for
        self.wk = nn.ModuleList([nn.Linear(self.dims[-1], self.dims[-1]) for i in range(options["time_steps"])])

    def forward(self, x, hidden):
        # Initialize an empty array to hold samples: (time steps, batch size, feature dimension)
        encoder_samples = th.empty((self.ts, self.bs, self.dims[-1])).float()
        # Initialize an empty array for predictions: (time steps, batch size, feature dimension)
        predictions = th.empty((self.ts, self.bs, self.dims[-1])).float()
        # Sample time point to use as anchor
        time_sample = th.randint(low=0, high=self.hindex, size=(1,)).long()
        # --- Forward pass
        # Encode input. Shape of (batch size, channel, length)
        z = self.encoder(x)
        # Reshape z to (batch size, length, channel) to be used with GRU()
        z = z.transpose(1, 2)
        # Extract encoder samples. Shape of z is (batch size, channel, length)
        for k in range(1, self.ts):
            # Shape of extracted tensor: (batch size, channel)
            encoder_samples[k-1] = z[:, time_sample + k, :].view(-1, self.dims[-1])
        # Get all data points in sequence prior to sampled time. Extracted tensor: (batch size, length, channel)
        prior_sequence = z[:, :time_sample+1, :]
        # Feed prior_sequence and hidden variable to generate output:o and hidden: h
        output, hidden = self.gru(prior_sequence, hidden)
        # Extract context from the output. Reshape it so that dimension of ct is (batch size, # of hidden units)
        ct = output[:, time_sample, :].view(self.bs, -1)
        # Generate predictions using context and linear weights per each time step
        for k in range(self.ts):
            # Shape of predictions[k]: (batch size, output dimension of wk[k] layer)
            predictions[k] = self.wk[k](ct)
        # Return samples of encoder for future times steps, predictions generated from GRU contexts, and hidden variable
        return encoder_samples, predictions, hidden

class CPCEncoder(nn.Module):
    """
    :param dict options: Generic dictionary to configure the model for training.
    :return: output (z) and hidden (h).

    Encoder is used for contrastive predictive coding.
    """
    def __init__(self, options):
        super(CPCEncoder, self).__init__()
        # Container to hold layers of the architecture in order
        self.layers = nn.ModuleList()
        # Get configuration that contains architecture and hyper-parameters
        self.options = copy.deepcopy(options)
        # Get the dimensions of all layers
        dims = options["conv_dims"]
        # Input image size. Example: 28 for a 28x28 image.
        img_size = self.options["img_size"]
        # Get dimensions for convolution layers in the following format: [i, o, k, s, p, d]
        # i=input channel, o=output channel, k = kernel size, s = stride, p = padding, d = dilation
        convolution_layers = dims[:-1]
        # Go through convolutional layers
        for layer_dims in convolution_layers:
            i, o, k, s, p, d = layer_dims
            self.layers.append(nn.Conv1d(i, o, k, stride=s, padding=p, dilation=d))
            # BatchNorm if True
            if options["isBatchNorm"]:
                self.layers.append(nn.BatchNorm1d(o))
            # Add activation
            self.layers.append(nn.LeakyReLU(inplace=False))
            # Dropout if True
            if options["isDropout"]:
                self.layers.append(nn.Dropout(options["dropout_rate"]))

    def forward(self, x):
        # Forward pass on convolutional layers: Expects the shape of input to be (batch size, channel, sequence length)
        for layer in self.layers:
            x = layer(x)
        return x