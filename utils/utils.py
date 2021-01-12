"""
Common utility functions.
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
Description: Utility functions
"""

import os
import sys
import yaml
import numpy as np
from numpy.random import seed
import random as python_random


def set_seed(options):
    seed(options["seed"])
    np.random.seed(options["seed"])
    python_random.seed(options["seed"])

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def set_dirs(config):
    """
    It sets up directory that will be used to save results.
    Directory structure:
          results -> evaluation
                  -> training -> model_mode -> loss
                                            -> model
                                            -> plots
    :return: None
    """
    # Update the config file with model config and flatten runtime config
    config = update_config_with_model(config)
    # Set main results directory using database name. Exp:  processed_data/dpp19
    paths = config["paths"]
    # data > processed_data
    processed_data_dir = os.path.join(paths["data"], "processed_data")
    # results > training
    training_dir = os.path.join(paths["results"], "training")
    # results > evaluation
    evaluation_dir = os.path.join(paths["results"], "evaluation")
    # results > training > model_mode = vae
    model_mode_dir = os.path.join(training_dir, config["model_mode"])
    # results > training > model_mode > model
    training_model_dir = os.path.join(model_mode_dir, "model")
    # results > training > model_mode > plots
    training_plot_dir = os.path.join(model_mode_dir, "plots")
    # results > training > model_mode > loss
    training_loss_dir = os.path.join(model_mode_dir, "loss")
    # Create any missing directories
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    if not os.path.exists(training_model_dir):
        os.makedirs(training_model_dir)
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
    if not os.path.exists(model_mode_dir):
        os.makedirs(model_mode_dir)
    if not os.path.exists(training_model_dir):
        os.makedirs(training_model_dir)
    if not os.path.exists(training_plot_dir):
        os.makedirs(training_plot_dir)
    if not os.path.exists(training_loss_dir):
        os.makedirs(training_loss_dir)
    # Print a message.
    print("Directories are set.")



def get_runtime_and_model_config():
    try:
        with open("./config/runtime.yaml", "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        sys.exit("Error reading runtime config file")
    # Update the config by adding the model specific config to runtime config
    config = update_config_with_model(config)
    return config


def update_config_with_model(config):
    model_config = config["unsupervised"]["model_mode"]
    try:
        with open("./config/"+model_config+".yaml", "r") as file:
            model_config = yaml.safe_load(file)
    except Exception as e:
        sys.exit("Error reading model config file")
    config.update(model_config)
    # TODO: Clean up structure of configs
    # Add sub-category "unsupervised" as a flat hierarchy to the config:
    config.update(config["unsupervised"])
    return config


def update_config_with_model_dims(data_loader, config):
    ((xi, xj), _) = next(iter(data_loader))
    # Get the number of features
    dim = xi.shape[-1]
    # Update the dims of model architecture by adding the number of features as the first dimension
    config["dims"].insert(0, dim)
    return config

