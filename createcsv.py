from scipy.misc import imread
import numpy as np
import pandas as pd
import os

root = './gestures'

# for each directory in the root folder
for directory, subdirectories, files in os.walk(root):
    # go through each file
    for file in files:
        # read the image file and extract its pixels

        # display the file name
        print(file)

        # read the file (image data)
        im = imread(os.path.join(directory, file))

        # flatten the image to 1 D array
        value = im.flatten()

        # I renamed the folders containing digits to the contained digit itself.
        # For example, digit_0 folder was renamed to 0.
        # so taking the 9th value of the folder gave the digit (i.e. "./train/8" ==> 9th value is 8),
        # which was inserted into the first column of the dataset.
        value = np.hstack((directory[8:], value))
        # transpose of DataFrame
        df = pd.DataFrame(value).T
        # shuffle the dataset
        # Return a random sample of items from an axis of object
        # frac - fraction of objects to be returned
        df = df.sample(frac=1)

        # open train_foo.csv in append and insert the flattened image pixel values
        with open('train_foo.csv', 'a') as dataset:
            df.to_csv(dataset, header=False, index=False)
