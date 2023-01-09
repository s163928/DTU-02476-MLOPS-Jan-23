# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import TensorDataset


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    logger.info("reading data")

    # Test data
    test_data = np.load(input_filepath + "/test.npz")

    # Train data
    file_list = [
        input_filepath + "/train_0.npz",
        input_filepath + "/train_1.npz",
        input_filepath + "/train_2.npz",
        input_filepath + "/train_3.npz",
        input_filepath + "/train_4.npz",
    ]
    data_all = [np.load(fname) for fname in file_list]

    logger.info("processing data")
    # Test data
    test_images = torch.from_numpy(test_data["images"])
    test_labels = torch.from_numpy(test_data["labels"])

    # Train data
    merged_train_data = {}
    for data in data_all:
        [merged_train_data.update({k: v}) for k, v in data.items()]

    train_images = torch.from_numpy(merged_train_data["images"])
    train_labels = torch.from_numpy(merged_train_data["labels"])

    test_images = (test_images - test_images.mean()) / test_images.std()
    train_images = (train_images - train_images.mean()) / train_images.std()

    test = TensorDataset(test_images, test_labels)
    train = TensorDataset(train_images, train_labels)
    processed_data = {"test": test, "train": train}

    logger.info("saving processed data")
    torch.save(processed_data, output_filepath + "/processed_data.pth")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
