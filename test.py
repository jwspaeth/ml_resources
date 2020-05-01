#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from config.config_handler import config_handler
import main
import datasets

# Questions

# Other feedback
#   • Increase size of some figures, to read axes and labels

if __name__ == "__main__":
    dataset = datasets.InfantDataset()
    dataset.feature_names = ["left_wrist_x", "left_wrist_y", "left_wrist_z"]
    dataset.offset = 6
    dataset.train_subject_names = ["k3"]
    training_data = dataset.load_data()

    print("Train ins shape: {}".format(training_data["train"]["ins"].shape))
    print("Train outs shape: {}".format(training_data["train"]["outs"].shape))

    for i in range(training_data["train"]["ins"].shape[0]):
        plt.plot(training_data["train"]["ins"][i, :, 0])

    plt.show()
