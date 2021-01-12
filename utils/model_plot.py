"""
Various plotting functionality to be used during training and evaluation.
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
"""

import matplotlib.pyplot as plt

def save_loss_plot(losses, plots_path):
    x_axis = list(range(len(losses["tloss_e"])))
    plt.plot(x_axis, losses["tloss_e"], c='r')
    title = "Training"
    if len(losses["vloss_e"]) >= 1:
        # If validation loss is recorded less often, we need to adjust x-axis values by the factor of difference
        beta = len(losses["tloss_e"]) / len(losses["vloss_e"])
        x_axis = list(range(len(losses["vloss_e"])))
        # Adjust the values of x-axis by beta factor
        x_axis = [beta*i for i in x_axis]
        plt.plot(x_axis, losses["vloss_e"], c='b')
        title += " and Validation "
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title + " Loss", fontsize=12)
    plt.tight_layout()
    plt.savefig(plots_path + "/loss.png")
