# Maths packages
import numpy as np

# Dataset packages
import pandas as pd

# Import os packages
from os import listdir
from os.path import isfile, join
import re

# Import progress bar
from tqdm import tqdm


def readData(filename):
    """Return the data of the file filename and save them as a numpy array."""

    # Resulting dct
    dct = {}

    with open(filename) as f:

        # Read the first line
        R, C, L, H = f.readline().split()

        # Append dct
        dct["R"] = R
        dct["C"] = C
        dct["L"] = L
        dct["H"] = H

        # Resulting dct
        dct["Grid"] = []

        # Read the other lines and save them in
        for line in f:
            dct["Grid"].append(line.strip())

    # Convert dct["Data"] as a numpy array
    dct["Grid"] = np.array(dct["Grid"])

    return dct


def readFiles(folder):
    """Read the data in folder and save them in a dict."""

    # Resulting dict
    result_dct = {}

    # List of files in folder
    files = [f for f in listdir(folder) if isfile(join(folder, f))]

    # Loop over all the files in folder
    for file in files:

        # Read the data and save them in dct
        file_name = re.sub("\...", "", file)
        result_dct[file_name] = readData(join(folder, file))

    return result_dct


def prediction(dataset_dct):
    """Make prediction for the current dataset_dct."""

    # Resulting array
    results = []

    # Extract R, C, L, H, data
    R = int(dataset_dct["R"])
    C = int(dataset_dct["C"])
    L = int(dataset_dct["L"])
    H = int(dataset_dct["H"])
    grid = dataset_dct["Grid"]

    return results


def predictionsDct(data_dct):
    """Extract the data of each file in dct and compute the predictions."""

    # Resulting dct
    predictions_dct = {}

    # Loop over the different datasets
    for key in data_dct.keys():

        # Compute the predictions for the current datasets
        predictions_dct[key] = prediction(data_dct[key])

    # Return the predictions made
    return predictions_dct


def writePredictions(predictions_dct, folder="Results/"):
    """Save the predictions."""

    # Loop over all the predictions save in array_dct
    for key in predictions_dct.keys():

        # Extract predictions for the given files
        slices = predictions_dct[key]

        # Count the number of slices
        nb_slices = len(slices)

        # Writes the result in a txt file
        f = open(folder + key + ".txt", "w")

        # Write number of slices
        f.write(str(nb_slices) + "\n")

        # Loop over each slices
        for i in slices:
            f.write(i + "\n")

        # Closing the file
        f.close()
