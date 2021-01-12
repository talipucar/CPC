"""
Library of loss functions.
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
"""

import numpy as np
import torch as th


def cpc_loss(encoder_samples, predictions, options):
    """
    :param encoder_samples: Output of CNN-based encoder
    :param predictions: Output generated from context, ct i.w. Wk*ct
    :param options: Dictionary that holds parameters
    :return:
    """
    # Number of time steps predicted
    time_steps = options["time_steps"]
    # Batch size
    bs = options["batch_size"]
    # Initialize softmax to compute accuracy
    softmax = th.nn.Softmax()
    # Initialize log-Softmax to compute loss
    log_softmax = th.nn.LogSoftmax()
    # Initialize loss
    InfoNCE = 0
    # Go through each time step, for which we made a prediction and accumulate loss and accuracy.
    for i in np.arange(0, time_steps):
        # Compute attention between encoder samples and predictions
        attention = th.mm(encoder_samples[i], th.transpose(predictions[i], 0, 1))
        # Correct classifications are those diagonal elements which has the highest attention in the column they are in.
        accuracy = th.sum(th.eq(th.argmax(softmax(attention), dim=0), th.arange(0, bs)))
        # InfoNCE is computed using log_softmax, and summing diagonal elements.
        InfoNCE += th.sum(th.diag(log_softmax(attention)))  # nce is a tensor
    # Negate and take average of InfoNCE over batch and time steps.
    # Minimizing -InfoNCE is equivalent to max attention along diagonal elements
    InfoNCE /= -1. * bs * time_steps
    # Compute the mean accuracy
    accuracy = 1. * accuracy.item() / bs
    # Return values
    return InfoNCE, accuracy