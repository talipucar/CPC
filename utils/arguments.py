"""
Setting arguments and configuration to run training and evaluation. Arguments can be provided via command line.
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
"""

import os
import pprint
import torch as th
from argparse import ArgumentParser
from os.path import dirname, abspath
from utils.utils import get_runtime_and_model_config

def get_arguments():
    # Initialize parser
    parser = ArgumentParser()
    # Dataset can be provided via command line
    parser.add_argument("-d", "--dataset", type=str, default="LibriSpeech")
    # Input image size
    parser.add_argument("-img", "--image_size", type=int, default=96)
    # Input channel size
    parser.add_argument("-ch", "--channel_size", type=int, default=3)
    # GPU device number as in "cuda:0". Defaul is 0.
    parser.add_argument("-cuda", "--cuda_number", type=str, default='0')
    # Experiment number if MLFlow is on
    parser.add_argument("-ex", "--experiment", type=int, default=1)
    # Return parser arguments
    return parser.parse_args()

def get_config(args):
    # Get path to the root
    root_path = dirname(abspath(__file__))
    # Get path to the runtime config file
    config = os.path.join(root_path, "config", "runtime.yaml")
    # Load runtime config from config folder: ./config/
    config = get_runtime_and_model_config()
    # Copy models argument to config to use later
    config["dataset"] = args.dataset
    # Copy image size argument to config to use later
    config["img_size"] = args.image_size
    # Copy channel size argument to config to modify default architecture in model config
    config["conv_dims"][0][0] = args.channel_size
    # Define which device to use: GPU or CPU
    config["device"] = th.device('cuda:'+args.cuda_number if th.cuda.is_available() else 'cpu')
    # Return
    return config

def print_config_summary(config, args):
    # Summarize config on the screen as a sanity check
    print(f"Here is the config being used:\n")
    pprint.pprint(config)
    print(f"Argument being used:\n")
    pprint.pprint(args)
    print(100*"=")