"""
Main function for training the model in Self-Supervised Learning setting.
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
"""

import time
import mlflow
from src.model import Model
from utils.load_data import AudioLoader
from utils.arguments import get_arguments, get_config, print_config_summary
from utils.utils import set_dirs

def train(config, data_loader, save_weights=True):
    """
    :param dict config: Dictionary containing options.
    :param IterableDataset data_loader: Pytorch data loader.
    :param bool save_weights: Saves model if True.
    :return:

    Utility function for saving on one training fold.
    """
    # Instantiate model
    encoder = Model(config)
    # Start the clock to measure the training time
    start = time.process_time()
    # Fit the model to the data
    encoder.fit(data_loader)
    # Total time spent on training
    training_time = time.process_time() - start
    # Report the training time
    print(f"Training time:  {training_time//60} minutes, {training_time%60} seconds")
    # Save the model for future use
    _ = encoder.save_weights() if save_weights else None
    print("Done with training...")
    # Track results
    if config["mlflow"]:
        # Log config with mlflow
        mlflow.log_artifacts("./config", "config")
        # Log model and results with mlflow
        mlflow.log_artifacts("./results/training/" + config["model_mode"], "training_results")
        # log model
        mlflow.pytorch.log_model(encoder, "models")

def main(config):
    # Ser directories (or create if they don't exist)
    set_dirs(config)
    # Pre-process, save, and load data. Set "get_all_test" True to use all test data when reporting
    # validation loss (this might slow down training and might crash due to memory).
    data_loader = AudioLoader(config)
    # Start training and save model weights at the end
    train(config, data_loader, save_weights=True)

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

