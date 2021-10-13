"""
Library of funtions to transform and load data.
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import soundfile as sf
from torch.utils import data as datautil


class AudioLoader(object):
    def __init__(self, config, train=True, kwargs={}):
        # Get config
        self.config = config
        # Get mode. If true, it will be used get the transformation for training. Else, get transformation for test
        self.train = train
        # Get the name of the dataset to be used
        dataset_name = config["dataset"]
        # Set main results directory using database name. 
        paths = config["paths"]
        # data > dataset_name
        file_path = os.path.join(paths["data"], dataset_name.lower())
        # Create the directory if it is missing
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # Get the datasets
        train_dataset, test_dataset = self.get_dataset(file_path)
        # Get batch size.
        bs = config["batch_size"]
        # Set the loader for training set
        self.train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True, **kwargs)
        # Set the loader for test set
        self.test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, drop_last=True, **kwargs)

    def get_dataset(self, file_path):
        train_dataset = RawDataset(directory=file_path + "/train-clean-100", audio_window=20480)

        # test_dataset = RawDataset(directory=file_path + "/dev-clean", audio_window=20480)
        # Return
        return train_dataset, train_dataset


class RawDataset(datautil.Dataset):

    def __init__(self, directory, audio_window):

        self.audio_window = audio_window
        self.audiolist = []
        self.idx2file = {}
        self.files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(directory):
            for file in f:
                if '.flac' in file:
                    self.files.append(os.path.join(r, file))

        for idx, filepath in enumerate(self.files):
            self.idx2file[idx] = filepath

    def __len__(self):
        """Denotes the total number of utterances """
        return len(self.files)

    def __getitem__(self, index):
        filepath = self.idx2file[index]
        audiodata, samplerate = sf.read(filepath)
        utt_len = audiodata.shape[0]
        # get the index to read part of the utterance into memory
        index = np.random.randint(utt_len - self.audio_window + 1)
        return audiodata[index:index + self.audio_window]


class DataTransforms(object):
    """
    Generates two transformed samples (referred as position pairs) for each sample from the dataset.
    """

    def __init__(self, size, train=True, jitter_strength=1.0):
        # Image size
        self.size = size
        # Whether the mode is train or not
        self.train = train
        # Jitter strength to use for color
        js = jitter_strength
        # Define color jitter
        color_jitter = transforms.ColorJitter(0.8 * js, 0.8 * js, 0.8 * js, 0.2 * js)
        # Define transformations for training set
        train_transform = [
                transforms.RandomResizedCrop(size=self.size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        # Define transformations for validation and test sets
        test_transform = [
                transforms.Resize(size=self.size),
                transforms.ToTensor(),
            ]
        # Compose train transformation
        self.train_transform = transforms.Compose(train_transform)
        # Compose validation/test transformation
        self.test_transform = transforms.Compose(test_transform)

    def __call__(self, x):
        # Get transformation to be used, based on the mode = (train, or evaluation/test)
        transform = self.train_transform if self.train else self.test_transform
        # Get two transformed samples from original sample
        xi, xj = transform(x), transform(x)
        # Return samples
        return xi, xj


class Loader(object):
    def __init__(self, config, download=True, get_all_train=False, get_all_test=False, train=True, kwargs={}):
        # Get config
        self.config = config
        # Get mode. If true, it will be used get the transformation for training. Else, get transformation for test
        self.train = train
        # Get the name of the dataset to be used
        dataset_name = config["dataset"]
        # Create dictionary to define number of classes in each dataset
        num_class = {'STL10': 10, 'CIFAR10': 10, 'MNIST': 10}
        # Set main results directory using database name. 
        paths = config["paths"]
        # data > dataset_name
        file_path = os.path.join(paths["data"], dataset_name.lower())
        # Create the directory if it is missing
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # Get the datasets
        train_dataset, test_dataset = self.get_dataset(dataset_name, file_path, download)
        # Get training batch size. If "get_all_train" is True, we get all training examples, else it is defined batch size.
        train_batch_size = train_dataset.data.shape[0] if get_all_train else config["batch_size"]
        # Get test batch size. If "get_all_test" is True, we get all test examples, else it is defined batch size.
        test_batch_size = test_dataset.data.shape[0] if get_all_test else config["batch_size"]
        # Set the loader for training set
        self.train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True, **kwargs)
        # Set the loader for test set
        self.test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=True, **kwargs)
        # Extract one batch to get image shape
        ((xi, _), _) = self.train_loader.__iter__().__next__()
        # Image shape after removing first dimension (which is batch size)
        self.img_shape = list(xi.size())[1:]
        # Number of classes in the dataset
        self.num_class = num_class[dataset_name]

    def get_dataset(self, dataset_name, file_path, download):
        # If True, use unlabelled data in addition to labelled training data
        train_split = 'train+unlabeled' if self.config["unlabelled_data"] else 'train'
        # Get train transformation
        train_transform = DataTransforms(size=self.config["img_size"], train=self.train)
        # Get test transformation
        test_transform = DataTransforms(size=self.config["img_size"], train=False)
        # Create dictionary for loading functions of datasets
        loader_map = {'STL10': datasets.STL10, 'CIFAR10': datasets.CIFAR10, 'MNIST': datasets.MNIST}
        # Get dataset
        dataset = loader_map[dataset_name]
        # Get datasets - Note that STL10 has a different input sets such as "split=" rather than "train="
        if dataset_name in ['STL10']:
            # Training and Validation datasets
            train_dataset = dataset(file_path, split=train_split, download=download, transform=train_transform)
            # Test dataset
            test_dataset = dataset(file_path, split='test', download=download, transform=test_transform)
        else:
            # Training and Validation datasets
            train_dataset = dataset(file_path, train=True, download=download, transform=train_transform)
            # Test dataset
            test_dataset = dataset(file_path, train=False, download=download, transform=test_transform)
        # Return
        return train_dataset, test_dataset

