"""
Evaluation of the model and baseline performance
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
"""

from tqdm import tqdm
import mlflow
from src.model import Model
from utils.load_data import AudioLoader
from utils.arguments import print_config_summary
from utils.arguments import get_arguments, get_config
from utils.utils import set_dirs
import torch as th


def eval(config):
    """
    :param dict config: Generic dictionary to configure the model for training.
    :return: None
    """
    # Don't use unlabelled data in train loader
    config["unlabelled_data"] = False
    # Load data.
    data_loader = AudioLoader(config)
    # Print which dataset we are using
    print(f"{config['dataset']} is being used to test performance.")
    # Get the performance using contrastive encoder
    model_performance(data_loader, config, baseline=False)
    # Get the baseline performance
    model_performance(data_loader, config, baseline=True)

def model_performance(data_loader, config, baseline=False):
    # Instantiate model
    model = Model(config)
    # Load model if we are not testing baseline performance. Baseline = Performance of randomly initialized model.
    if not baseline:
        model.load_models()
    # Change the mode to evaluation
    model.set_mode("evaluation")
    # Get cpc model
    cpc = model.cpc
    # Validation dataset
    validation_loader = data_loader.test_loader
    # Compute total number of batches per epoch
    total_batches = len(validation_loader)
    print(f"Total number of samples / batches in validation set: {len(validation_loader.dataset)} / {len(validation_loader)}")
    # Attach progress bar to data_loader to check it during validation. "leave=True" gives a new line per epoch
    val_tqdm = tqdm(enumerate(validation_loader), total=total_batches, leave=True)
    # Initialize accuracy
    accuracy = 0
    # Go through batches
    for i, Xbatch in val_tqdm:
        # Add channel dim to Xbatch, and turn it into a tensor
        Xbatch = model.process_batch(Xbatch)
        # Initialize first hidden layer to zeros
        hidden = th.zeros(2, config["batch_size"], config["conv_dims"][-1])
        # Get encoder samples, predictions and final hidden layer
        encoder_samples, predictions, hidden = cpc(Xbatch, hidden)
        # Get accuracy
        accuracy = accuracy + get_accuracy(encoder_samples, predictions, config)
    # Compute mean accuracy across all batches
    accuracy = accuracy/total_batches
    print(f"Mean accuracy for validation set: {accuracy}")


def get_accuracy(encoder_samples, predictions, config):
    time_steps = config["time_steps"]
    # Batch size
    bs = config["batch_size"]
    # Initialize softmax to compute accuracy
    softmax = th.nn.Softmax()
    # Initialize log-Softmax to compute loss
    log_softmax = th.nn.LogSoftmax()
    # Initialize loss
    InfoNCE = 0
    # Go through each time step, for which we made a prediction and accumulate loss and accuracy.
    for i in range(0, time_steps):
        # Compute attention between encoder samples and predictions
        attention = th.mm(encoder_samples[i], th.transpose(predictions[i], 0, 1))
        # Correct classifications are those diagonal elements which has the highest attention in the column they are in.
        accuracy = th.sum(th.eq(th.argmax(softmax(attention), dim=0), th.arange(0, bs)))
    # Compute the mean accuracy
    accuracy = 1. * accuracy.item() / bs
    return accuracy


def main(config):
    # Ser directories (or create if they don't exist)
    set_dirs(config)
    # Start training
    eval(config)


if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    config = get_config(args)
    # Summarize config and arguments on the screen as a sanity check
    print_config_summary(config, args)
    # --If True, start of MLFlow for experiment tracking:
    if config["mlflow"]:
        # Experiment name
        mlflow.set_experiment(experiment_name=config["model_mode"]+"_"+str(args.experiment))
        # Start a new mlflow run
        with mlflow.start_run():
            # Run the main
            main(config)
    else:
        # Run the main
        main(config)