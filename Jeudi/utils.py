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
        N = f.readline().split()

        # Append dct
        dct["N"] = N

        # Resulting dataFrame
        df = pd.DataFrame(columns=["ID", "H_V", "M", "Tags"])

        # Resulting convertion table
        table = {}

        # Counter
        counter = 0

        # Read the other lines and save them in
        for i, line in enumerate(f):

            # Extract info
            string = line.strip().split()
            infos = [i, string[0], int(string[1]), " ".join(string[2:])]

            # Extract tags
            tags = string[2:]
            for tag in tags:

                # Test if present in table
                if not(tag in table):
                    table[tag] = counter
                    counter += 1

            # Add the infos to df
            df.loc[i] = infos

        # Add df to dct and table
        dct["Photos"] = df
        dct["Table"] = table

    return dct


def readFiles(folder):
    """Read the data in folder and save them in a dict."""

    # Resulting dict
    result_dct = {}

    # List of files in folder
    files = [f for f in listdir(folder) if isfile(join(folder, f))]

    # Loop over all the files in folder
    for file in tqdm(files):

        # Read the data and save them in dct
        file_name = re.sub("\...", "", file)
        result_dct[file_name] = readData(join(folder, file))

    return result_dct


# Load files
data_dct = np.load('./Save/data_dct.npy').item()
test_dct = np.load('./Save/test_dct.npy').item()

# Display test_dct
data_dct["a_examplet"]


# data_dct = readFiles("Data")

def convertTags(tags_str, table):
    """Convert a string of tags into a string of integer trough table."""

    # Split str
    tags_l = tags_str.split()

    # Loop of convertion
    tags_int_l = [str(table[tag]) for tag in tags_l]

    # Convert as a string
    result = " ".join(tags_int_l)

    return result

def convertAsInt(dct):
    """Convert tag as int."""

    for key in tqdm(dct.keys()):

        # Extract df and table of dct
        df = dct[key]["Photos"]
        table = dct[key]["Table"]

        # Definition of the lambda function
        convert = lambda tags_str : convertTags(tags_str, table)

        # Add a colum with integers
        df["Tags_Int"] = df["Tags"].apply(lambda x: convert(x))

        # Update df of dct
        dct[key] = df

    return dct

# Save the dictionnary
# np.save('./Save/data_dct.npy', data_dct)
# np.save('./Save/test_dct.npy', test_dct)

# Load
data_dct = np.load('./Save/data_dct.npy').item()
# test_dct = np.load('./Save/test_dct.npy').item()

# Display test_dct
# data_dct["a_examplet"]


def prediction(df):
    """Make prediction for the current dataset_dct."""

    # Resulting array
    results = []

    # Extract N
    N = len(df)

    # Predictions
    for i in range(len(df)):

        if df.iloc[i,:]["H_V"] != "V":
            results.append([df.iloc[i,:]["ID"]])

    return results


def predictionsDct(data_dct):
    """Extract the data of each file in dct and compute the predictions."""

    # Resulting dct
    predictions_dct = {}

    # Loop over the different datasets
    for key in tqdm(data_dct.keys()):

        # Compute the predictions for the current datasets
        predictions_dct[key] = prediction(data_dct[key])

    # Return the predictions made
    return predictions_dct

# Compute the predictions
# predictions_dct = predictionsDct(data_dct)

def writePredictions(predictions_dct, folder="Results/"):
    """Save the predictions."""

    # Loop over all the predictions save in array_dct
    for key in predictions_dct.keys():

        # Extract predictions for the given files
        photos = predictions_dct[key]

        # Count the number of photos in the slideshow
        nb_photos = len(photos)

        # Writes the result in a txt file
        f = open(folder + key + ".txt", "w")

        # Write number of slices
        f.write(str(nb_photos) + "\n")

        # Loop over each slices
        for ID in photos:

            # Convert ID as str
            ID_str = [str(i) for i in ID]

            # Write IDs
            f.write(" ".join(ID_str) + "\n")

        # Closing the file
        f.close()

# Load
# predictions_dct = np.load('./Save/predictions_dct.npy').item()
# writePredictions(predictions_dct)
